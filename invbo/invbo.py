import torch
import gpytorch
import math
import selfies as sf 
import torch.nn.functional as F

from gpytorch.mlls import PredictiveLogLikelihood 
from invbo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from invbo.utils.utils import update_models_end_to_end, update_surr_model
from invbo.utils.bo_utils.ppgpr import GPModelDKL
from invbo.utils.mol_utils.selfies_vae.data import collate_fn

class InvBOState:

    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        k=50,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=5,
        acq_func='ts',
        verbose=True,
        alpha=10.0, # Lip loss
        beta=1, # Surr loss
        gamma=1, # VAE loss
        delta=1,
        ):

        self.objective          = objective         # objective with vae for particular task
        self.train_x            = train_x           # initial train x data
        self.train_y            = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.acq_func           = acq_func          # acquisition function: Thompson Sampling (ts)
        self.verbose            = verbose
        self.alpha              = alpha
        self.beta               = beta
        self.gamma              = gamma
        self.delta              = delta
        
        assert acq_func in ["ts"]
        if minimize:
            self.train_y = self.train_y * -1

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()

    def initialize_xs_to_scores_dict(self,):
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_top_k(self):
        ''' Initialize top k x, y, and zs'''
        self.top_k_scores, top_k_idxs = torch.topk(self.train_y.squeeze(), min(self.k, len(self.train_y)))
        self.top_k_scores = self.top_k_scores.tolist()
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs = [self.train_x[i] for i in top_k_idxs]
        self.top_k_zs = self.train_z[top_k_idxs]

    def initialize_tr_state(self):
        self.tr_state = TurboState(
            dim=self.train_z.shape[-1],
            batch_size=self.bsz, 
            best_value=torch.max(self.train_y).item(),
            failure_tolerance=4
            )

        return self

    def initialize_surrogate_model(self ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        n_pts = min(self.train_z.shape[0], 1024)
        self.model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood).cuda()
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))
        self.model = self.model.eval() 
        self.model = self.model.cuda()

        return self

    def update_next(self, z_next_, y_next_, x_next_, acquisition=False): 
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
            and update trust region state
        '''
        z_next_ = z_next_.detach().cpu() 
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)

        progress = False
        skip_idx = []
        for i, score in enumerate(y_next_):
            if x_next_[i] in self.train_x:
                skip_idx.append(i)
                continue
                
            self.train_x.append(x_next_[i])
            if len(self.top_k_scores) < self.k: 
                self.top_k_scores.append(score.item())
                self.top_k_xs.append(x_next_[i])
                self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                self.top_k_zs = torch.cat((self.top_k_zs, z_next_[i].unsqueeze(0)))
            elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):
                # if the score is better than the worst score in the top k list, upate the list
                min_score = min(self.top_k_scores)
                min_idx = self.top_k_scores.index(min_score)
                self.top_k_scores[min_idx] = score.item()
                self.top_k_xs[min_idx] = x_next_[i]
                self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2) # .cuda()
            #if we imporve
            if score.item() > self.best_score_seen:
                self.progress_fails_since_last_e2e = 0
                progress = True
                self.best_score_seen = score.item() #update best
                self.best_x_seen = x_next_[i]
                self.new_best_found = True
        if (not progress) and acquisition: # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(state=self.tr_state, Y_next=y_next_)
    
        for i in range(len(z_next_)):
            if i not in skip_idx:
                self.train_z = torch.cat((self.train_z, z_next_[i].unsqueeze(0)), dim=-2)
                self.train_y = torch.cat((self.train_y, y_next_[i].unsqueeze(0)), dim=-2)

        return self

    def update_surrogate_model(self): 
        if not self.initial_model_training_complete:
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
        else:
            n_epochs = self.num_update_epochs
            new_zs = self.train_z[-self.bsz:]
            new_ys = self.train_y[-self.bsz:].squeeze(-1).tolist()
            train_z = torch.cat((new_zs, self.top_k_zs))
            train_y = torch.tensor(new_ys + self.top_k_scores).float()               
            
        self.model = update_surr_model(
            self.model,
            self.mll,
            self.learning_rte,
            train_z,
            train_y,
            n_epochs,
        )
        self.initial_model_training_complete = True

        return self

    def update_models_e2e(self, track_with_wandb, tracker):
        '''Finetune VAE end to end with surrogate model'''
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.bsz:]
        new_ys = self.train_y[-self.bsz:].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()
        self.objective, self.model = update_models_end_to_end(
            train_x,
            train_y,
            self.objective,
            self.model,
            self.mll,
            self.learning_rte,
            self.num_update_epochs,
            track_with_wandb,
            tracker,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
        )
        self.tot_num_e2e_updates += 1

        return self

    def acquisition(self):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        z_next = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=self.top_k_zs,
            Y=torch.tensor(self.top_k_scores),
            batch_size=self.bsz, 
            acqf=self.acq_func,
        )
        with torch.no_grad():
            out_dict = self.objective(z_next)
            z_next = out_dict['valid_zs']
            y_next = out_dict['scores']
            x_next = out_dict['decoded_xs']       
            if self.minimize:
                y_next = y_next * -1

        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(
                z_next,
                y_next,
                x_next,
                acquisition=True
            )
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")

    def inversion(self):
        self.objective.vae.eval()
        new_xs = self.train_x[-self.bsz:]
        train_x = new_xs + self.top_k_xs

        bsz=64
        num_batches = math.ceil(len(train_x) / bsz) 

        init_zs = torch.zeros([0]).cuda()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx, stop_idx = batch_idx*bsz, (batch_idx+1)*bsz
                xs_batch = train_x[start_idx:stop_idx]

                X_list = []
                for smile in xs_batch:
                    try:
                        selfie = self.objective.smiles_to_selfies[smile]
                    except:
                        selfie = sf.encoder(smile)
                        self.objective.smiles_to_selfies[smile] = selfie
                    tokenized_selfie = self.objective.dataobj.tokenize_selfies([selfie])[0]
                    encoded_selfie = self.objective.dataobj.encode(tokenized_selfie).unsqueeze(0)
                    X_list.append(encoded_selfie)
                tokens = collate_fn(X_list)    
                z, _ = self.objective.vae.encode(tokens.cuda())
                init_zs = torch.cat((init_zs, z), dim=0)                

        init_zs.requires_grad_()

        num_e = 1000
        final_z = torch.zeros_like(init_zs)
        finish_idx = []
        for batch_idx in range(num_batches):
            optimizer = torch.optim.Adam([
                {'params': init_zs, 'lr': 1e-1}
            ])
            start_idx, stop_idx = batch_idx*bsz, (batch_idx+1)*bsz
            input_z = init_zs[start_idx:stop_idx].cuda()            
            config = train_x[start_idx:stop_idx]

            X_list = []
            for smile in config:
                try:
                    selfie = self.objective.smiles_to_selfies[smile]
                except:
                    selfie = sf.encoder(smile)
                    self.objective.smiles_to_selfies[smile] = selfie
                tokenized_selfie = self.objective.dataobj.tokenize_selfies([selfie])[0]
                encoded_selfie = self.objective.dataobj.encode(tokenized_selfie).unsqueeze(0)
                X_list.append(encoded_selfie)
            X = collate_fn(X_list)
        
            for e in range(num_e):
                optimizer.zero_grad()
                self.objective.vae.zero_grad()

                logits = self.objective.vae.decode(input_z, X.cuda())
                recon_loss = F.cross_entropy(logits.permute(0, 2, 1), X.cuda())

                mean_acc = (logits.argmax(dim=-1) == X.cuda()).float().mean(-1)
                stop_idx = torch.where(mean_acc == 1)[0]
                if len(stop_idx) != 0:
                    remove_idx = []
                    for idx in stop_idx:
                        if (idx.item() + batch_idx*bsz) not in finish_idx:
                            finish_idx.append(idx.item() + batch_idx*bsz)
                            remove_idx.append(idx.item() + batch_idx*bsz)
                    final_z[remove_idx] = init_zs[remove_idx].detach()

                if (1 - (logits.argmax(dim=-1) == X.cuda()).float().mean()) < 1e-9:
                    break

                loss = recon_loss
                loss.backward()
                optimizer.step()

        non_one_idx = torch.tensor([i for i in [*range(len(init_zs))] if i not in finish_idx])
        if len(non_one_idx) != 0:
            final_z[non_one_idx] = init_zs[non_one_idx].detach()
        
        self.train_z[-self.bsz:] = final_z.reshape(final_z.shape[0], -1)[:self.bsz].cpu()
        self.top_k_zs = final_z.reshape(final_z.shape[0], -1)[-len(self.top_k_zs):].cpu()

        return self                