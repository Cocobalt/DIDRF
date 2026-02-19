import numpy as np
import matplotlib.pyplot as plt
import os

class AdaptiveControlMonitor:

    def __init__(self, log_dir="./adaptive_control_logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.iterations = []
        self.unfairness_values = []
        self.param_values = {} 
        self.ndcg_values = []
        
        self.pid_outputs = []
        self.optimizer_outputs = []
        self.kalman_outputs = []
        self.lqr_outputs = []
        self.final_params = []
    
    def record(self, iteration, unfairness, ndcg, pid_output, optimizer_output, 
               kalman_output, lqr_output, final_param):
        self.iterations.append(iteration)
        self.unfairness_values.append(unfairness)
        self.ndcg_values.append(ndcg)
        self.pid_outputs.append(pid_output)
        self.optimizer_outputs.append(optimizer_output)
        self.kalman_outputs.append(kalman_output)
        self.lqr_outputs.append(lqr_output)
        self.final_params.append(final_param)
    
    def plot_performance(self, save=True):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.iterations, self.unfairness_values)
        plt.title('Unfairness over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Unfairness')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.iterations, self.ndcg_values)
        plt.title('NDCG over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('NDCG')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.iterations, self.pid_outputs, label='PID')
        plt.plot(self.iterations, self.optimizer_outputs, label='Optimizer')
        plt.plot(self.iterations, self.kalman_outputs, label='Kalman')
        plt.plot(self.iterations, self.lqr_outputs, label='LQR')
        plt.title('Controller Outputs over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.iterations, self.final_params)
        plt.title('Final Adaptive Parameter over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Parameter Value')
        plt.grid(True)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.log_dir, 'performance_metrics.png'))
            plt.close()
        else:
            plt.show()
    
    def save_data(self):
        data = {
            'iterations': self.iterations,
            'unfairness': self.unfairness_values,
            'ndcg': self.ndcg_values,
            'pid_outputs': self.pid_outputs,
            'optimizer_outputs': self.optimizer_outputs,
            'kalman_outputs': self.kalman_outputs,
            'lqr_outputs': self.lqr_outputs,
            'final_params': self.final_params
        }
        
        np.save(os.path.join(self.log_dir, 'adaptive_control_data.npy'), data)
        
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.log_dir, 'adaptive_control_data.csv'), index=False)
        
        return data
