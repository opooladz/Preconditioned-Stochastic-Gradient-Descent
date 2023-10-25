
class SGD_(Optimizer):

    def __init__(self, params_with_grad, lr_params=0.01):
        # mutable members
        self.lr_params = lr_params

        # protected members
        params_with_grad = [params_with_grad, ] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]  # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._delta_param_scale = torch.finfo(dtype).eps ** 0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]

		self._d = 1e-6
        self._s = 0
        self._z = None
        self._x0 = None
        self._sgs = 0 


    @torch.no_grad()
    def step(self, closure):


        # only evaluates the gradients
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = torch.autograd.grad(loss, self._params_with_grad)


        
        # flatten gradients
        grad = torch.cat([torch.flatten(g) for g in grads])

        dlr = self.lr_params * self._d /torch.linalg.vector_norm(grad,ord=2)
        
        if self._z is None:
            self._s = torch.zeros_like(grad).detach()   
            self._z = self._x0 = torch.clone(torch.cat([torch.flatten(g) for g in self._params_with_grad])).detach()       
        else:
          self._s = self._s + dlr*grad
          self._z = self._z - dlr*grad


        for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes):
          param.data = 0.9*param.data + 0.1*self._z[j - i:j].view_as(param)
          # param.requires_grad_(True)

        self._sgs+=(dlr*torch.dot(grad,self._s))
        # print(torch.linalg.vector_norm(self._s,ord=2))
        d_hat = 2*self._sgs/torch.linalg.vector_norm(self._s,ord=2)**2
        self._d = max(self._d,d_hat)
        print(dlr)
        print(self._d)
        # print(self._d)

        # return whatever closure returns
        return closure_returns
