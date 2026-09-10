import logging
import math
import torch
import numpy as np
from typing import Literal

class Trainer(object):
    def __init__(self, 
        model, 
        optimizer, 
        rho, 
        lossfunc=torch.nn.MSELoss(reduction='mean'), 
        max_norm=0.1, 
        angular_reg_weight=0.0, 
        output_reg_weight=0.0, 
        exponent_reg_weight=0.0,
        log_file = "results.log",
        scheduler = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.rho = rho
        self.scheduler = scheduler
        self.angular_reg_weight = angular_reg_weight
        self.exponent_reg_weight = exponent_reg_weight
        self.output_reg_weight = output_reg_weight
        self.max_norm = max_norm
        self.body_order = self.model.angular_coupling.body_order
        self.epoch = 0

        self.logfile_update(log_file)
    
    def logfile_update(self, log_file):

        self.log_file = log_file

        logger = logging.getLogger("Trainer")
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger


    def train(self, dataloader, device):
        loss_total = 0
        loss_total_ene = 0
        loss_total_for = 0
        loss_total_angular_reg = 0
        loss_total_exponent_reg = 0
        loss_total_output_reg = 0
        total_mse_energy = 0
        total_mse_force = 0
        nan_flag = 0
        n_samples = 0
        self.model.train()
        for batch in dataloader:
            batch = batch.to(device)
            batch=batch.to_dict()
            self.optimizer.zero_grad()
            output = self.model(batch)
            natoms = (batch["ptr"][1:]-batch["ptr"][0:-1]).view(-1,1)
            natoms_sum = torch.sum(natoms)
            lossene = self.lossfunc(output["total_energy"].view(-1,1)/natoms, batch["total_energy"].view(-1,1)/natoms)
            total_mse_energy += lossene.item()
            lossene *= self.rho
            lossfor = self.lossfunc(output["forces"], batch["forces"])
            loss = lossene + lossfor        
            if self.angular_reg_weight > 0:
                lossang = self.model.bij.compute_l2loss()
                lossang = self.angular_reg_weight*lossang
                loss_total_angular_reg += lossang.item()
                loss += lossang
            if self.output_reg_weight > 0:
                lossout = torch.sum(torch.pow(self.model.bij.linear.weight,2))
                lossout = self.output_reg_weight*lossout
                loss_total_output_reg += lossout.item()
                loss += lossout
            if self.exponent_reg_weight > 0:
                exponent = self.model.bij.extpow.coeff*self.model.bij.extpow.smoothclamp(self.model.bij.extpow.invexponent) + 1.0
                lossexp = exponent.pow(2).sum()
                lossexp = self.exponent_reg_weight*lossexp
                loss_total_exponent_reg += lossexp.item()
                loss += lossexp
            loss.backward()
            loss_total += loss.item()
            loss_total_ene += lossene.item()
            loss_total_for += lossfor.item()
            total_mse_force = loss_total_for
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            if math.isnan(loss.item()):
                nan_flag = 1
                torch.save({'model_state_dict': self.model.state_dict()}, 'modelnan.pth')
                logging.error("NaN detected! Model saved to modelnan.pth")
                with open("param_nan.txt", 'a') as f:
                    f.write(str(list(self.model.named_parameters())))
                break
            n_samples += 1
            self.optimizer.step()    
        if nan_flag == 1:
            return np.nan,np.nan,np.nan
        
        rmse_energy = math.sqrt(total_mse_energy / n_samples)
        rmse_force = math.sqrt(total_mse_force / n_samples)
        return loss_total, loss_total_angular_reg, loss_total_output_reg, loss_total_exponent_reg, loss_total_ene, loss_total_for, rmse_energy, rmse_force

    def valid(self, dataloader, device):
        loss_total = 0
        loss_total_ene = 0
        loss_total_for = 0
        total_mse_energy = 0
        total_mse_force = 0
        nan_flag = 0
        n_samples = 0
        self.model.eval()
  
        for batch in dataloader:
            batch = batch.to(device)
            batch=batch.to_dict()
            output = self.model(batch)
            natoms = (batch["ptr"][1:]-batch["ptr"][0:-1]).view(-1,1)
            natoms_sum = torch.sum(natoms)
            lossene = self.lossfunc(output["total_energy"].view(-1,1)/natoms, batch["total_energy"].view(-1,1)/natoms)
            total_mse_energy += lossene.item()
            lossene *= self.rho
            lossfor = self.lossfunc(output["forces"], batch["forces"])
            loss = lossene + lossfor
            loss_total += loss.item()
            loss_total_ene += lossene.item()
            loss_total_for += lossfor.item()
            total_mse_force = loss_total_for
            n_samples += 1
        rmse_energy = math.sqrt(total_mse_energy / n_samples)
        rmse_force = math.sqrt(total_mse_force / n_samples)
        return loss_total, loss_total_ene, loss_total_for, rmse_energy, rmse_force

    def run(self, nepoch, train_loader, device, valid_args = None, filename_best = None, stop_lr = None, best_mode = 'train', best_valid_label = None, scheduler_criterion = 'train', criterion_valid_label = None):

        if valid_args is not None:
            valid_labels = valid_args['labels']
            valid_loaders = valid_args['loaders']
            valid_intervals = valid_args['intervals']
            assert len(valid_labels) == len(valid_loaders)
            assert len(valid_intervals) == len(valid_loaders)
        
        loss_best = np.inf

        for epoch in range(nepoch):
            self.epoch += 1
            loss_train,loss_angular_reg,loss_output_reg,loss_exponent_reg,loss_ene,loss_for,rmse_energy,rmse_force=self.train(train_loader, device)
            if math.isnan(loss_train):
                print('Loss is NaN!')
                break
            log_message = (
                f"Epoch #{self.epoch:<4}: "
                f"RMSE Energy: {rmse_energy*1000:.2f} | "
                f"RMSE Force: {rmse_force*1000:.2f} | "
                f"lr: {self.optimizer.param_groups[0]["lr"]:.2e} | "
                f"Loss Total: {loss_train:.4e} | "
                f"Loss Ang: {loss_angular_reg:.4e} | "
                f"Loss Out: {loss_output_reg:.4e} | "
                f"Loss Exp: {loss_exponent_reg:.4e} | "
                f"Loss Ene: {loss_ene:.4e} | "
                f"Loss For: {loss_for:.4e} | "
            )

            if self.scheduler is not None:
                if scheduler_criterion == 'train':
                    self.scheduler.step(loss_train)
            
            if best_mode == 'train':
                if loss_train < loss_best:
                    loss_best = loss_train
                    torch.save(self.model.state_dict(), filename_best)
                    log_message += f"Best | "

            self.logger.info(log_message)

            if valid_args is not None:
                for valid_label, valid_loader, valid_interval in zip(valid_labels, valid_loaders, valid_intervals):
                    if self.epoch%valid_interval == 0:
                        loss_valid,loss_ene,loss_for,rmse_energy,rmse_force=self.valid(valid_loader, device)
                        log_message = (
                            f"{valid_label}: "
                            f"RMSE Energy: {rmse_energy*1000:.2f} | "
                            f"RMSE Force: {rmse_force*1000:.2f} | "
                            f"lr: {self.optimizer.param_groups[0]["lr"]:.2e} | "
                            f"Loss Total: {loss_valid:.4e} | "
                        )
                    
                        if best_mode == 'valid':
                            if best_valid_label is None:
                                if valid_label == valid_labels[0]:
                                    if loss_valid < loss_best:
                                        loss_best = loss_valid
                                        torch.save(self.model.state_dict(), filename_best)
                                        log_message += f"Best | "
                            else:
                                if valid_label == best_valid_label:
                                    if loss_valid < loss_best:
                                        loss_best = loss_valid
                                        torch.save(self.model.state_dict(), filename_best)
                                        log_message += f"Best | "
                        
                        self.logger.info(log_message)

                        if self.scheduler is not None:
                            if scheduler_criterion == 'valid':
                                if criterion_valid_label is None:
                                    if valid_label == valid_labels[0]:
                                        self.scheduler.step(loss_valid)
                                else:
                                    if valid_label == criterion_valid_label:
                                        self.scheduler.step(loss_valid)
            
            if self.optimizer.param_groups[0]["lr"] <= stop_lr:
                break
        
        self.logger.info("Finished!")
            




    
    