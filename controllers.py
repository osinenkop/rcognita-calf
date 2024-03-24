"""
Contains controllers a.k.a. agents.

"""

from utilities import dss_sim
from utilities import rep_mat
from utilities import uptria2vec
from utilities import push_vec
import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from scipy.stats import multivariate_normal
from numpy.linalg import lstsq
from numpy import reshape
import warnings
import math
# For debugging purposes
from tabulate import tabulate
import os

def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, mode):
    """
    Main interface for various controllers.

    Parameters
    ----------
    mode : : string
        Controller mode as acronym of the respective control method.

    Returns
    -------
    action : : array of shape ``[dim_input, ]``.
        Control action.

    """
    
    if mode=='manual': 
        action = action_manual
    elif mode=='nominal': 
        action = ctrl_nominal.compute_action(t, observation)
    else: # Controller for benchmakring
        action = ctrl_benchmarking.compute_action(t, observation)
        
    return action


class ControllerOptimalPredictive:
    """
    Class of predictive optimal controllers, primarily model-predictive control and predictive reinforcement learning, that optimize a finite-horizon cost.
    
    Currently, the actor model is trivial: an action is generated directly without additional policy parameters.
        
    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled.
    mode : : string
        Controller mode. Currently available (:math:`\\rho` is the running objective, :math:`\\gamma` is the discounting factor):
          
        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1
    
           * - Mode
             - Cost function
           * - 'MPC' - Model-predictive control (MPC)
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
           * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
             - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
           * - 'SQL' - RL/ADP via stacked Q-learning
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\hat \\gamma^{k-1} Q^{\\theta}(y_{N_a}, u_{N_a})`               
        
        Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
        
        *Add your specification into the table when customizing the agent*.    

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default).
    action_init : : array of shape ``[dim_input, ]``   
        Initial action to initialize optimizers.          
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).
    Nactor : : natural number
        Size of prediction horizon :math:`N_a`. 
    pred_step_size : : number
        Prediction step size in :math:`J_a` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon.
    sys_rhs, sys_out : : functions        
        Functions that represent the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
    buffer_size : : natural number
        Size of the buffer to store data.
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of running objectives along horizon.
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
    critic_period : : number
        The critic is updated every ``critic_period`` units of time. 
    critic_struct : : natural number
        Choice of the structure of the critic's features.
        
        Currently available:
            
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quad-lin'
             - Quadratic-linear
           * - 'quadratic'
             - Quadratic
           * - 'quad-nomix'
             - Quadratic, no mixed terms
           * - 'quad-mix'
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
               where :math:`w` is the critic's weight vector
       
        *Add your specification into the table when customizing the critic*. 
    run_obj_struct : : string
        Choice of the running objective structure.
        
        Currently available:
           
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quadratic'
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars`` should be ``[R1]``
           * - 'biquadratic'
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars``
               should be ``[R1, R2]``   
        
        *Pass correct run objective parameters in* ``run_obj_pars`` *(as a list)*
        
        *When customizing the running objective, add your specification into the table above*
        
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        
        
    """       
    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 buffer_size=20,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 run_obj_struct='quadratic',
                 run_obj_pars=[],
                 observation_target=[],
                 state_init=[],
                 obstacle=[],
                 seed=1):
        """
            Parameters
            ----------
            dim_input, dim_output : : integer
                Dimension of input and output which should comply with the system-to-be-controlled.
            mode : : string
                Controller mode. Currently available (:math:`\\rho` is the running objective, :math:`\\gamma` is the discounting factor):
                
                .. list-table:: Controller modes
                :widths: 75 25
                :header-rows: 1
            
                * - Mode
                    - Cost function
                * - 'MPC' - Model-predictive control (MPC)
                    - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
                * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
                    - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
                * - 'SQL' - RL/ADP via stacked Q-learning
                    - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`               
                
                Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
                
                *Add your specification into the table when customizing the agent* .   
        
            ctrl_bnds : : array of shape ``[dim_input, 2]``
                Box control constraints.
                First element in each row is the lower bound, the second - the upper bound.
                If empty, control is unconstrained (default).
            action_init : : array of shape ``[dim_input, ]``   
                Initial action to initialize optimizers.              
            t0 : : number
                Initial value of the controller's internal clock
            sampling_time : : number
                Controller's sampling time (in seconds)
            Nactor : : natural number
                Size of prediction horizon :math:`N_a` 
            pred_step_size : : number
                Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
                convenience. Larger prediction step size leads to longer factual horizon.
            sys_rhs, sys_out : : functions        
                Functions that represent the right-hand side, resp., the output of the exogenously passed model.
                The latter could be, for instance, the true model of the system.
                In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
                Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
            buffer_size : : natural number
                Size of the buffer to store data.
            gamma : : number in (0, 1]
                Discounting factor.
                Characterizes fading of running objectives along horizon.
            Ncritic : : natural number
                Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
                optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
            critic_period : : number
                The critic is updated every ``critic_period`` units of time. 
            critic_struct : : natural number
                Choice of the structure of the critic's features.
                
                Currently available:
                    
                .. list-table:: Critic feature structures
                :widths: 10 90
                :header-rows: 1
            
                * - Mode
                    - Structure
                * - 'quad-lin'
                    - Quadratic-linear
                * - 'quadratic'
                    - Quadratic
                * - 'quad-nomix'
                    - Quadratic, no mixed terms
                * - 'quad-mix'
                    - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
                    where :math:`w` is the critic's weights
            
                *Add your specification into the table when customizing the critic*.
            run_obj_struct : : string
                Choice of the running objective structure.
                
                Currently available:
                
                .. list-table:: Running objective structures
                :widths: 10 90
                :header-rows: 1
            
                * - Mode
                    - Structure
                * - 'quadratic'
                    - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars`` should be ``[R1]``
                * - 'biquadratic'
                    - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars``
                    should be ``[R1, R2]``
            """        

        np.random.seed(seed)
        print(seed)

        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        self.action_sqn_init = []
        self.state_init = []

        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
            self.action_init = self.action_min/10
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)
        
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )        
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys   
        
        # Learning-related things
        self.buffer_size = buffer_size
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.run_obj_struct = run_obj_struct
        self.run_obj_pars = run_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0
        print('---Critic structure---', self.critic_struct)

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input) ) 
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 )
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_output + self.dim_output * self.dim_input + self.dim_input )
            self.Wmin = -1e3*np.ones(self.dim_critic)  
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'poly3':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input ) )
            self.Wmin = -1e3*np.ones(self.dim_critic)  
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'poly4':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 * 3)
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = np.ones(self.dim_critic) 


        self.w_critic_init = np.random.uniform(10, 1000, size = self.dim_critic)
        
        with np.printoptions(precision=2, suppress=True):
            print("---Critic initial weights---", self.w_critic_init)
            
        ##############################################################################
        ################################################################## CALF things
        self.w_critic_LG = self.w_critic_init
        self.observation_LG = state_init
        self.action_LG = self.action_sqn_init

        self.w_critic_init_LG = self.w_critic_init

        self.w_critic_buffer_LG = []
        self.N_CTRL = N_CTRL(ctrl_bnds)

        if self.mode == "SARSA-m":
            self.coef_for_alpha_low = 1e-2 
            self.coef_for_alpha_up = 1e4 
            self.nu = 1e-4 
        elif self.mode == "CALF":
            self.coef_for_alpha_low = 1e-1
            self.coef_for_alpha_up = 1e3
            self.nu = 1e-5 

        self.critic_init_fading_factor = 0.8

        self.action_buffer_LG = np.zeros( [self.buffer_size, self.dim_input] )
        self.observation_buffer_LG = np.zeros( [self.buffer_size, self.dim_output] )   

        self.state_init = state_init

        self.count_CALF = 0    
        self.count_N_CTRL = 0   

        self.debug=True
        self.delta_w_factor = 0.5
        self.circle_x, self.circle_y, self.sigma = obstacle[0], obstacle[1], obstacle[2]

        self.accum_obj_val_LG =1e20
      

    def reset(self,t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock, value and current actions are reset.
        All the learned parameters are retained.
        
        """        

        # Controller: common

        if len(self.action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
            self.action_init = self.action_min/10
        else:
            self.action_curr = self.action_init
            self.action_sqn_init = rep_mat( self.action_init , 1, self.Nactor)
        
        self.action_buffer = np.zeros( [self.buffer_size, self.dim_input] )
        self.observation_buffer = np.zeros( [self.buffer_size, self.dim_output] )        

        self.critic_clock = t0
        self.ctrl_clock = t0

##############################################################################  Reset last good buffers
        self.observation_LG = self.state_init
        self.action_LG = self.action_sqn_init

        self.action_buffer_LG = np.zeros( [self.buffer_size, self.dim_input] )
        self.observation_buffer_LG = np.zeros( [self.buffer_size, self.dim_output] )   

        self.count_CALF = 0    
        self.count_N_CTRL = 0   
############################################################
        total_sum = 0
        N = len(self.w_critic_buffer_LG)  
        for i, w_i in enumerate(self.w_critic_buffer_LG, start=0):
            total_sum += (self.critic_init_fading_factor ** i) * w_i
        if N != 0:
            weighted_average = total_sum / N
        else:
            weighted_average = self.w_critic_init

        Delta_w =  weighted_average - self.w_critic_init

# Weights update
############################################################
        if self.mode == "CALF":
        # Uncomment to activate initial weight update based on a cost improvement criterion
            if self.accum_obj_val <= self.accum_obj_val_LG: 
                print(f"Final cost:    {self.accum_obj_val:.2f}" + f"    Best cost:    {self.accum_obj_val_LG:.2f}")
                self.accum_obj_val_LG = self.accum_obj_val
                self.w_critic_init_LG = self.w_critic_init
                self.w_critic_init = self.w_critic_init + self.delta_w_factor * Delta_w
            else:
                # self.w_critic_init = self.w_critic_init_LG
                self.w_critic_init = self.w_critic_init_LG + self.delta_w_factor * np.clip( self.Wmax/5 * np.random.normal(size=self.dim_critic), self.Wmin, self.Wmax )
        elif self.mode == "SARSA-m":
            print(f"Final cost:    {self.accum_obj_val:.2f}" + f"    Best cost:    {self.accum_obj_val_LG:.2f}")
            self.accum_obj_val_LG = self.accum_obj_val
            self.w_critic_init_LG = self.w_critic_init
            self.w_critic_init = self.w_critic_init + self.delta_w_factor * Delta_w           

        self.w_critic_LG = self.w_critic_init
        self.accum_obj_val = 0
        self.w_critic_buffer_LG = []

    def get_CALF_count(self):
        return self.count_CALF

    def get_N_CTRL_count(self):
        return self.count_N_CTRL
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def run_obj(self, observation, action):
        """
        Running (equivalently, instantaneous or stage) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        run_obj = 0

        def obstacle_penalty(observation, obstacle_positions, sigma, penalty_factor):
            """
            Calculates the value of probability density function of a bivariate normal distribution at a given point.
            Arguments:
            x, y : float
                Coordinates of the point at which to calculate the probability density value.
            mu_x, mu_y : float
                Mean values (expectations) along the X and Y axes, respectively.
            sigma_x, sigma_y : float
                Standard deviations along the X and Y axes, respectively.
            rho : float
                Correlation coefficient between X and Y.

            Returns:
            float
                Value of the probability density function of a bivariate normal distribution at the given point (x, y).
            """
            mu_x = obstacle_positions[0]
            sigma_x = sigma

            mu_y = obstacle_positions[1]
            sigma_y = sigma
            rho = 0
            x = observation[0]
            y = observation[1]
            z = ((x - mu_x) ** 2) / (sigma_x ** 2) + ((y - mu_y) ** 2) / (sigma_y ** 2) - (2 * rho * (x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
            denom = 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)
            return np.exp(-z / (2 * (1 - rho ** 2))) / denom * penalty_factor
        

        obstacle_positions = np.array([self.circle_x, self.circle_y])
        penalty = obstacle_penalty(observation, obstacle_positions, self.sigma, penalty_factor=1e1)

        if self.run_obj_struct == 'quadratic':
            R1 = self.run_obj_pars[0]
            run_obj = chi @ R1 @ chi + penalty
        elif self.run_obj_struct == 'biquadratic':
            R1 = self.run_obj_pars[0]
            R2 = self.run_obj_pars[1]
            run_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return run_obj
        
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) running objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        self.accum_obj_val += self.run_obj(observation, action)*self.sampling_time
  
    def _critic(self, observation, action, w_critic):
        """
        Critic a.k.a. objective learner: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi
        elif self.critic_struct == 'quad-mix':
            regressor_critic = np.concatenate([ observation**2, np.kron(observation, action), action**2 ]) 
        elif self.critic_struct == 'poly3':
            regressor_critic = np.concatenate([uptria2vec( np.outer(chi**2, chi) ), uptria2vec( np.outer(chi, chi) )])
        elif self.critic_struct == 'poly4':               
            regressor_critic = np.concatenate([uptria2vec( np.outer(chi**2, chi**2) ), uptria2vec( np.outer(chi**2, chi) ), uptria2vec( np.outer(chi, chi) )])

        return w_critic @ regressor_critic
    
    def _critic_cost(self, w_critic):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        Jc = 0
        
        for k in range(self.Ncritic-1, 0, -1):
            observation_prev = self.observation_buffer[k-1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k-1, :]
            action_next = self.action_buffer[k, :]
            
            # Temporal difference
            
            critic_prev = self._critic(observation_prev, action_prev, w_critic)
            critic_next = self._critic(observation_next, action_next, self.w_critic_LG)
            e = critic_prev - self.gamma * critic_next - self.run_obj(observation_prev, action_prev)
            
            Jc += 1/2 * e**2 
            
        # With penalty on critic weights    
        # return Jc + 0.95 * np.linalg.norm(w_critic - self.w_critic_LG)**2
        return Jc
             
    def _critic_optimizer(self, observation=[], action=[], do_update=True):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.ControllerOptimalPredictive._critic_cost`.

        """        

        def CALF_constr_1(observation, action, w_critic):
            critic_new = self._critic(observation, action, w_critic)                            # Q^w (x_k, u_k)
            critic_LG = self._critic(self.observation_LG, self.action_LG, self.w_critic_LG)     # Q^w°(x°, u°)
            return critic_new - critic_LG                                                       # Q^w (x_k, u_k) - Q^w°(x°, u°)
        
        def CALF_constr_2(observation, action, w_critic):
            return self._critic(observation, action, w_critic)                                  # alpha_low(||x_k||) <= Q^w (x_k, u_k) <= alpha_up(||x_k||)

        final_constraints = []
        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 40, 'disp': False} 
        else:
            critic_opt_options = {'maxiter': 40, 'maxfev': 80, 'disp': False, 'adaptive': True, 'xatol': 1e-3, 'fatol': 1e-3} # 'disp': True, 'verbose': 2} 
        
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)

        if self.mode == "SARSA-m":
            max_desirable_CALF_decay_rate = 0.1
            final_constraints.append(sp.optimize.NonlinearConstraint(lambda w_critic: CALF_constr_1(observation=observation, action=action, w_critic=w_critic), 
                                                                     -max_desirable_CALF_decay_rate * self.sampling_time, -self.nu * self.sampling_time))
            final_constraints.append(sp.optimize.NonlinearConstraint(lambda w_critic: CALF_constr_2(observation=observation, action=action, w_critic=w_critic), 
                                                                     self.coef_for_alpha_low*np.linalg.norm(observation)**2, self.coef_for_alpha_up*np.linalg.norm(observation)**2))
            if do_update:
                print(f"Updating critic at {self.critic_clock:.2f} s")
                w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_init, method=critic_opt_method, tol=1e-3, bounds=bnds, options=critic_opt_options).x    
            else:
                w_critic = self.w_critic_LG

            self.constr_2 = self._critic(observation, action, w_critic)                                       # Qw
            self.constr_3 = self._critic(self.observation_LG, self.action_LG, self.w_critic_LG)               # Qw° 
            self.constr_1 = self.constr_2  - self.constr_3                                                    # Qw - Qw°
        else:
            w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_init, method=critic_opt_method, tol=1e-3, bounds=bnds, options=critic_opt_options).x

        if self.mode == "CALF":

            final_constraints.append(sp.optimize.NonlinearConstraint(lambda w_critic: CALF_constr_1(observation=observation, action=action, w_critic=w_critic), 
                                                                     -np.inf, -self.nu * self.sampling_time))
            final_constraints.append(sp.optimize.NonlinearConstraint(lambda w_critic: CALF_constr_2(observation=observation, action=action, w_critic=w_critic), 
                                                                     self.coef_for_alpha_low*np.linalg.norm(observation)**2, self.coef_for_alpha_up*np.linalg.norm(observation)**2))
            if do_update:
                print(f"Updating critic at {self.critic_clock:.2f} s")
                w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_LG, method=critic_opt_method, tol=1e-3, bounds=bnds, 
                                    constraints=final_constraints, options=critic_opt_options).x  
            else:
                w_critic = self.w_critic_LG
                
            self.constr_2 = self._critic(observation, action, w_critic)                                       # Qw
            self.constr_3 = self._critic(self.observation_LG, self.action_LG, self.w_critic_LG)               # Qw° 
            self.constr_1 = self.constr_2  - self.constr_3                                                    # Qw - Qw°
        else:
            w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_init, method=critic_opt_method, tol=1e-3, bounds=bnds, options=critic_opt_options).x        
        
        return w_critic
    
    def _actor_cost(self, action_sqn, observation):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """
        
        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        
        # System observation prediction
        observation_sqn[0, :] = observation
        state = self.state_sys
        for k in range(1, self.Nactor):
            state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k-1, :])  # Euler scheme
            
            observation_sqn[k, :] = self.sys_out(state)
        
        J = 0         
        if self.mode=='MPC':
            for k in range(self.Nactor):
                J += self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :])
        elif self.mode=='RQL':     # RL: Q-learning with Ncritic-1 roll-outs of running objectives
             for k in range(self.Nactor-1):
                J += self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :])
             J += self._critic(observation_sqn[-1, :], my_action_sqn[-1, :], self.w_critic)
        elif self.mode=='SQL':     # RL: stacked Q-learning
             for k in range(self.Nactor): 
                Q = self._critic(observation_sqn[k, :], my_action_sqn[k, :], self.w_critic)
        elif self.mode == "CALF" or self.mode == "SARSA-m":
             for k in range(self.Nactor):                
                Q = self._critic(observation_sqn[k, :], my_action_sqn[k, :], self.w_critic_LG) 
                J += Q 

        return J
    
    def _actor_optimizer(self, observation):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.ControllerOptimalPredictive._actor_cost`.
        See class documentation.
        
        Customization
        -------------         
        
        This method normally should not be altered, adjust :func:`~controllers.ControllerOptimalPredictive._actor_cost` instead.
        The only customization you might want here is regarding the optimization algorithm.

        

        # For direct implementation of state constraints, this needs `partial` from `functools`
        # See [here](https://stackoverflow.com/questions/27659235/adding-multiple-constraints-to-scipy-minimize-autogenerate-constraint-dictionar)
        # def state_constraint(action_sqn, idx):
            
        #     my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])
            
        #     observation_sqn = np.zeros([idx, self.dim_output])    
            
        #     # System output prediction
        #     if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
        #         observation_sqn[0, :] = observation
        #         state = self.state_sys
        #         Y[0, :] = observation
        #         x = self.x_s
        #         for k in range(1, idx):
        #             # state = get_next_state(state, my_action_sqn[k-1, :], delta)
        #             state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #             observation_sqn[k, :] = self.sys_out(state)            
            
        #     return observation_sqn[-1, 1] - 1

        # my_constraints=[]
        # for my_idx in range(1, self.Nactor+1):
        #     my_constraints.append({'type': 'eq', 'fun': lambda action_sqn: state_constraint(action_sqn, idx=my_idx)})

        # my_constraints = {'type': 'ineq', 'fun': state_constraint}

        # Optimization method of actor    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        # actor_opt_method = 'SLSQP' # Standard
        """
        
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 40, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 40, 'maxfev': 60, 'disp': False, 'adaptive': True, 'xatol': 1e-3, 'fatol': 1e-3}
       
        isGlobOpt = 0
        
        my_action_sqn_init = np.reshape(self.action_sqn_init, [self.Nactor*self.dim_input,])
        
        bnds = sp.optimize.Bounds(self.action_sqn_min, self.action_sqn_max, keep_feasible=True)
        
        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-3, 'options': actor_opt_options}
                action_sqn = basinhopping(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                          my_action_sqn_init,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter = 10).x
            else:
                action_sqn = minimize(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                      my_action_sqn_init,
                                      method=actor_opt_method,
                                      tol=1e-3,
                                      bounds=bnds,
                                      options=actor_opt_options).x        

        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            action_sqn = self.action_curr
        
        return action_sqn[:self.dim_input]    # Return first action
                    
    def compute_action(self, t, observation):
        """
        Main method. See class documentation.
        
        Customization
        -------------         
        
        Add your modes, that you introduced in :func:`~controllers.ControllerOptimalPredictive._actor_cost`, here.

        """       
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            if self.mode == 'MPC':  
                
                action = self._actor_optimizer(observation)
                    
            elif self.mode in ['RQL', 'SQL']:
                # Critic
                timeInCriticPeriod = t - self.critic_clock
                
                # Update data buffers
                self.action_buffer = push_vec(self.action_buffer, self.action_curr)
                self.observation_buffer = push_vec(self.observation_buffer, observation)
                
                if timeInCriticPeriod >= self.critic_period:
                    # Update critic's internal clock
                    self.critic_clock = t
                    
                    self.w_critic = self._critic_optimizer()
                    self.w_critic_prev = self.w_critic
                    
                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.w_critic_init = self.w_critic_prev
                    
                else:
                    self.w_critic = self.w_critic_prev
                    
                # Actor
                action = self._actor_optimizer(observation) 
                         
            elif self.mode == "CALF":
                self.observation_buffer = push_vec(self.observation_buffer, observation)
                self.action_buffer = push_vec(self.action_buffer, self.action_curr)

                if np.linalg.norm(observation[:2]) > 0.2:
                    action = self._actor_optimizer(observation)
                    
                    timeInCriticPeriod = t - self.critic_clock
                    if timeInCriticPeriod >= self.critic_period:
                        # Update critic's internal clock
                        self.critic_clock = t
                        
                    do_update = (timeInCriticPeriod >= self.critic_period)    
                    self.w_critic = self._critic_optimizer(observation, action, do_update=do_update)


                minus_nu_dt = -self.nu * self.sampling_time
                a_low = self.coef_for_alpha_low*np.linalg.norm(observation)**2
                a_up = self.coef_for_alpha_up*np.linalg.norm((observation))**2

                if (self.constr_1 <= minus_nu_dt
                        and 
                        a_low <= self.constr_2 <= a_up
                        and 
                        np.linalg.norm(observation[:2]) > 0.2):

                    self.w_critic_LG = self.w_critic
                    self.observation_LG = observation
                    self.action_LG = action

                    self.w_critic_buffer_LG.append(self.w_critic)
                    self.observation_buffer_LG = push_vec(self.observation_buffer_LG, observation)
                    self.action_buffer_LG = push_vec(self.action_buffer_LG, action)

                    self.count_CALF = self.count_CALF + 1
                    print("\033[93m") # Color it to indicate CALF
                else: 
                    print("\033[0m")
                    self.count_N_CTRL = self.count_N_CTRL + 1
                    action = self.N_CTRL.pure_loop(observation)

            elif self.mode == "SARSA-m":
                self.observation_buffer = push_vec(self.observation_buffer, observation)
                self.action_buffer = push_vec(self.action_buffer, self.action_curr)

                action = self._actor_optimizer(observation)
                
                timeInCriticPeriod = t - self.critic_clock
                if timeInCriticPeriod >= self.critic_period:
                    # Update critic's internal clock
                    self.critic_clock = t
                    
                do_update = (timeInCriticPeriod >= self.critic_period)    
                self.w_critic = self._critic_optimizer(observation, action, do_update=do_update)


                minus_nu_dt = -self.nu * self.sampling_time
                a_low = self.coef_for_alpha_low*np.linalg.norm(observation)**2
                a_up = self.coef_for_alpha_up*np.linalg.norm((observation))**2

                if (self.constr_1 <= minus_nu_dt
                        and 
                        a_low <= self.constr_2 <= a_up
                        and 
                        np.linalg.norm(observation[:2]) > 0.2) : # 13 line

                    self.w_critic_LG = self.w_critic
                    self.observation_LG = observation
                    self.action_LG = action

                    self.w_critic_buffer_LG.append(self.w_critic)
                    self.observation_buffer_LG = push_vec(self.observation_buffer_LG, observation)
                    self.action_buffer_LG = push_vec(self.action_buffer_LG, action)

                    self.count_CALF = self.count_CALF + 1
                    print("\033[93m") # Color it to indicate SARSA-m
                else: 
                    print("\033[0m")
                    self.count_N_CTRL = self.count_N_CTRL + 1

            elif self.mode == "N_CTRL":
                action = self.N_CTRL.pure_loop(observation)
            
            self.action_curr = action
            
            return action    
    
        else:
            return self.action_curr

class N_CTRL:

    def __init__(self, ctrl_bnds):

        self.K_l = 0.5
        self.k_a = 20
        self.k_b = -3.5

        self.v_min = ctrl_bnds[0, 0]
        self.v_max = ctrl_bnds[0, 1]
        self.omega_min = ctrl_bnds[1, 0]
        self.omega_max = ctrl_bnds[1, 1]

    def pure_loop(self, observation):

        x_start = observation[0]
        y_start = observation[1]
        rotation = observation[2]
        goal_x = 0
        goal_y = 0
        goal_z = 0

        error_x = goal_x - x_start
        error_y = goal_y - y_start

        error_rot_frame = -rotation + np.arctan2(error_y, error_x) 

        beta = -rotation - error_rot_frame

        dist2goal = math.sqrt(error_x**2 + error_y**2)
        
        v = self.K_l * dist2goal
        w = self.k_a * error_rot_frame + self.k_b * beta
        
        if -np.pi < error_rot_frame <= -np.pi/2 or np.pi/2 < error_rot_frame <= np.pi:
            v = -v

        if v < self.v_min:
            v = self.v_min
        elif v > self.v_max:
            v = self.v_max

        if w < self.omega_min:
            w = self.omega_min
        elif w > self.omega_max:
            w = self.omega_max

        return [v,w]



