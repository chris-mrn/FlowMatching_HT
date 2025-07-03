import numpy as np
import torch
# from marginalTailAdaptiveFlow.utils.flows import experiment
import flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from .utils.scaler_grad import NativeScalerWithGradNormCount as NativeScaler
from matplotlib import cm
import torch.nn as nn

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

class LightTailedModel(nn.Module):
    def __init__(
        self,PreVFnet,dimension,simulation=False
    ):
        # self.features = features
        super(LightTailedModel, self).__init__()
        self.PreVFnet=PreVFnet
        self.dimension=dimension
        self.sim=simulation
    def forward(self, x_t,time_t):
        dim=self.dimension
        if self.sim:

            time_t=time_t.reshape(-1).expand(x_t.shape[0])
        else:
            time_t=time_t.unsqueeze(1)


        prefinal_vf=self.PreVFnet(x_t,time_t)

        prefinal_vf=prefinal_vf.reshape(prefinal_vf.shape[0],-1)

        velocity_field=prefinal_vf

        return(velocity_field)

class HT_FlowMatching_X0:
    def __init__(self, tail_net, flow_net, TTF, dimension, device):
        self.model = flow_net
        self.Tail_paramNet = tail_net
        self.TTF=TTF
        self.device=device
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.dim=dimension
        self.flow_model = LightTailedModel(flow_net,dimension,simulation=False).to(device)

    def train_epoch(self,optim1,optim2,train_loader,noise_loader, epoch):
        loss_scaler = NativeScaler()
        iI = epoch
        for data, noise in zip(train_loader, noise_loader):

            optim1.zero_grad()
            optim2.zero_grad()
            const=-1
            param_tail_pre=self.Tail_paramNet(1+torch.zeros(noise.shape[0],1).to(self.device)) #BX80
            dummy_tail_param=param_tail_pre.reshape(param_tail_pre.shape[0],4,self.dim)
            _unc_pos_tail,_unc_neg_tail,shift,_unc_scale = dummy_tail_param[:,0,:]**2+const,dummy_tail_param[:,1,:]**2+const,dummy_tail_param[:,2,:]*0,dummy_tail_param[:,3,:]*0      # i am keeping shift 0 and var softplus(0)
            param_tail=torch.cat([_unc_pos_tail,_unc_neg_tail,shift,_unc_scale],1)

            x_1 = data.float().to(self.device)   #batch x 20
            x_0 = noise.float().to(self.device)#torch.randn_like(x_1).float().to(self.device) #batch x 20
            x_0 = self.TTF(x_0,param_tail)


            if iI<(3*self.epochs)//4:
                t = 1-torch.sqrt(1-torch.rand(x_1.shape[0])).to(self.device) #best
            else:
                t= -torch.log(1 - torch.rand(x_1.shape[0]) * (1 - torch.exp(torch.tensor(-1)))).to(self.device)

            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

            x_t,time_t,dx_t=path_sample.x_t,path_sample.t,path_sample.dx_t  #x_t- B X 20
            # print("CHECK GRAD",x_1.shape,x_t.requires_grad,dx_t.requires_grad)

            velocity_field=self.flow_model(x_t,time_t)

            loss = torch.pow( velocity_field - dx_t, 2).mean()


            loss_scaler(
                loss,
                optim1,
                optim2,
                parameters=self.model.parameters(),
                parameters2=self.Tail_paramNet.parameters(),
                update_grad=True,
                approach=2
                )
        print(f"Loss epoch {epoch}", loss.item())

    def train(self, optim1, optim2, train_loader, noise_loader, epochs):
        self.epochs = epochs
        for epoch in range(epochs):
            self.train_epoch(optim1,optim2,train_loader,noise_loader, epoch)

    def flow(self,x_t,t):
        return self.flow_model(x_t, t)


    def sample_from(self, X0):
        # step size for ode solver
        self.flow_model.sim=True
        wrapped_vf = WrappedModel(self.flow_model)
        step_size = 0.01
        const=-1
        param_tail_pre=self.Tail_paramNet(1+torch.zeros(X0.shape[0],1).to(self.device)) #BX80
        dummy_tail_param=param_tail_pre.reshape(param_tail_pre.shape[0],4,self.dim)
        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:]**2+const,dummy_tail_param[:,1,:]**2+const,dummy_tail_param[:,2,:]*0,dummy_tail_param[:,3,:]*0      # i am keeping shift 0 and var softplus(0)
        param_tail=torch.cat([_unc_pos_tail,_unc_neg_tail,shift,_unc_scale],1)

        T = torch.linspace(0,1,10)  # sample times
        T = T.to(self.device)

        x_init = X0
        x_init=self.TTF(x_init,param_tail)
        solver = ODESolver(velocity_model=wrapped_vf)
        sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)
        sol = sol.cpu().numpy()
        generated_data=sol[9]

        self.flow_model.sim=False


        return(generated_data)