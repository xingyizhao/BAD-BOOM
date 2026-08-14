import torch
from torch.optim import AdamW
from trl import SFTTrainer
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.opt.modeling_opt import OPTAttention

"""
Function (Main-Experiment: Attack-Optimizers): This script includes SAM and BAD-BOOM optimizers.

    AdamW: https://arxiv.org/pdf/1711.05101
    SAM: https://arxiv.org/pdf/2010.01412  

Developer: Xingyi Zhao. 
Update: 2026-08-13
Logan, Utah, USA
"""

############################################################################################################################
class SAM(torch.optim.Optimizer):
    ''' Sharpness-Aware Minimization (SAM) optimizer. Reference: https://arxiv.org/abs/2010.01412'''
    '''I modified the pytorch implementation from https://github.com/davda54/sam for distributed training in the trl trainer'''

    def __init__(self, params, base_opt_cls=AdamW, rho=0.01, **opt_kwargs):
        defaults = dict(**opt_kwargs)
        super().__init__(params, defaults)
        self.rho = rho
        self.base_opt = base_opt_cls(self.param_groups, **opt_kwargs)
    
    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        total = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None: 
                    total += p.grad.pow(2).sum()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return total.sqrt()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        scale = self.rho / (self._grad_norm() + 1e-7)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(scale * p.grad.to(p.device))
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        return self.base_opt.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        return self.base_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_opt.state_dict()

    def load_state_dict(self, state_dict):
        out = self.base_opt.load_state_dict(state_dict)
        self.param_groups = self.base_opt.param_groups
        return out

############################################################################################################################  
class BADBOOM(torch.optim.Optimizer):
    def __init__(self, params, base_opt_cls=AdamW, rho=0.01, **opt_kwargs):
        defaults = dict(**opt_kwargs)
        super().__init__(params, defaults)
        self.rho = rho
        self.base_opt = base_opt_cls(self.param_groups, **opt_kwargs)

    @torch.no_grad()
    def first_step(self, g_map, fisher_diag_map, zero_grad=False):
        device = self.param_groups[0]["params"][0].device
        denom_sqrt = torch.zeros(1, device=device)
        
        for group in self.param_groups:
            for p in group["params"]:
                if (p in g_map) and (p in fisher_diag_map):
                    gp = g_map[p]
                    Fp = fisher_diag_map[p]
                    denom_sqrt += (gp.pow(2) * Fp).sum()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(denom_sqrt, op=torch.distributed.ReduceOp.SUM)
        
        denom = torch.sqrt(denom_sqrt)
        scale = self.rho / (denom + 1e-7)

        for group in self.param_groups:
            for p in group["params"]:
                if (p in g_map) and (p in fisher_diag_map):
                    ellipsoid_w = scale * (fisher_diag_map[p] * g_map[p])
                    self.state[p]["old_p"] = p.data.clone()
                    p.add_(ellipsoid_w.to(device))

        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        return self.base_opt.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        return self.base_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_opt.state_dict()

    def load_state_dict(self, state_dict):
        out = self.base_opt.load_state_dict(state_dict)
        self.param_groups = self.base_opt.param_groups
        return out

############################################################################################################################ 
# Defending Fine-tuning Attack with Vaccine: https://arxiv.org/pdf/2402.01109   
def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
    # or isinstance(module, LlamaMLP)
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention):
            module_list+= [module]
    # # print(module_list)
    return module_list

class VaccineTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss)

            return loss 
        # print("calling sam")
        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.vaccine_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.vaccine_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(perturbation.shape)
            # print(output.shape)
            return output         
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        
        # print(self.vaccine_state["gradient"].items())
        grad_norm = self._grad_norm(self.vaccine_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.vaccine_state["gradient"]:
            # grad_norm = self._grad_norm(self.vaccine_state["gradient"][module])
            grad = self.vaccine_state["gradient"][module]
            scale = 0.001 / (grad_norm + 1e-7)  # rho
            e_r = (grad) * scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()
            # print(module)
        #     print( torch.norm(self.vaccine_state["e_r"][module]) )
        # print(len(self.vaccine_state["e_r"]))
    
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.vaccine_state["e_r"]:
        #     module.weight.data -= self.vaccine_state["e_r"][module]
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([

                    ( poison_grads_representation[name] ).norm(p=2)
      
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm