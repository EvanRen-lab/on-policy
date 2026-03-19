#!/usr/bin/env python
"""
DFL Training Main System - Supports multiple allocation strategies, includes resource allocation time recording.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import various allocators
try:
    from allocator_mappo import mappo_allocator
    MAPPO_AVAILABLE = True
    print("✓ MAPPO allocator available")
except ImportError:
    MAPPO_AVAILABLE = False
    print("⚠ MAPPO allocator not available")

try:
    from allocator_ao import alternating_allocator
    ALTERNATING_AVAILABLE = True
    print("✓ Alternating optimization allocator available")
except ImportError:
    ALTERNATING_AVAILABLE = False
    print("⚠ Alternating optimization allocator not available")

try:
    from allocator_avg import avg_allocator
    AVG_AVAILABLE = True
    print("✓ Average allocation allocator available")
except ImportError:
    AVG_AVAILABLE = False
    print("⚠ Average allocation allocator not available")

try:
    from allocator_random import random_allocator
    RANDOM_AVAILABLE = True
    print("✓ Random search allocator available")
except ImportError:
    RANDOM_AVAILABLE = False
    print("⚠ Random search allocator not available")

# Import custom modules
from models import ModelFactory
from datasets import DatasetFactory
from mobility import TrajectoryGenerator
from plot_utils import plot_results, plot_time_breakdown


class Terminal:
    """Terminal class containing model, data and training logic."""

    def __init__(self, terminal_id: int, config, dataloader=None):
        self.id = terminal_id
        self.config = config
        self.weight = config.weights[terminal_id]

        # Initialize model
        self.model = ModelFactory.create_model(
            config.model_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained
        ).to(config.device)

        self.initial_state_dict = self.model.state_dict().copy()

        # Optimizer
        if config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Learning rate scheduler
        if config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_decay_step,
                gamma=config.lr_decay_gamma
            )
        elif config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.communication_rounds
            )
        else:
            self.scheduler = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Data loader
        self.dataloader = dataloader

        # Training statistics
        self.train_loss_history = []
        self.train_acc_history = []
        self.communication_rounds = 0

        print(f"Terminal {terminal_id} initialized, weight = {self.weight:.3f}")

    def local_train(self, num_epochs=1):
        """Local training."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.config.device), target.to(self.config.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / (len(self.dataloader) * num_epochs)
        accuracy = 100.0 * correct / total

        self.train_loss_history.append(avg_loss)
        self.train_acc_history.append(accuracy)

        return avg_loss, accuracy

    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)

                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy

    def get_model_state(self):
        return self.model.state_dict().copy()

    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def reset_model(self):
        self.model.load_state_dict(self.initial_state_dict)

    def get_parameters(self):
        return [param.data.clone() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = new_param.clone()


class DFLTrainer:
    """DFL Trainer main class."""

    def __init__(self, config):
        self.config = config

        # Create directories
        os.makedirs(config.result_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize data partitioner
        self.partitioner = DatasetFactory.create_partitioner(config)

        # Load data
        print("\nLoading and partitioning data...")
        self.train_loaders, self.test_loader = self.partitioner.create_dataloaders()

        # Initialize terminals
        print("\nInitializing terminals...")
        self.terminals = [
            Terminal(i, config, self.train_loaders[i])
            for i in range(config.num_agents)
        ]

        # Initialize trajectory generator
        self.trajectory_generator = TrajectoryGenerator(config)

        # Training history
        self.history = {
            'communication_rounds': [],
            'weighted_accuracy': [],
            'simple_accuracy': [],
            'terminal_accuracies': [],
            'terminal_losses': [],
            'neighbor_counts': [],
            'positions': [],
            'connectivity_stats': [],
            'allocation_times': [],
            'allocation_strategy_times': {
                'mappo': 0.0, 'alternating': 0.0, 'avg': 0.0, 'random': 0.0,
                'distance': 0.0, 'fixed': 0.0
            }
        }

        # Communication statistics
        self.communication_stats = {
            'total_communications': 0,
            'total_bytes_transferred': 0,
            'total_allocation_time': 0.0,
            'avg_allocation_time': 0.0,
            'max_allocation_time': 0.0,
            'allocation_time_history': []
        }

        # Training time statistics
        self.training_times = {
            'total_training_time': 0.0,
            'local_training_times': [],
            'aggregation_times': [],
            'evaluation_times': [],
            'round_times': []
        }

        print(f"\nDFL Trainer initialized")
        print(f"Allocation strategy: {config.allocation_strategy}")
        print(f"Record allocation time: {config.record_allocation_time}")
        print(f"Number of agents: {config.num_agents}")
        print(f"Agent weights: {config.weights}")
        print(f"Data distribution: {config.data_distribution}")
        print(f"Communication mode: {config.communication_mode}")
        print(f"Communication rounds: {config.communication_rounds}")
        print(f"SINR threshold: {config.sinr_threshold_db} dB")

    def _get_wireless_config(self, round_idx):
        """Build wireless parameter dictionary for allocators."""
        return {
            'num_agents': self.config.num_agents,
            'num_channels': self.config.num_channels,
            'max_power': self.config.max_power,
            'max_power_dbm': self.config.max_power_dbm,
            'bandwidth': self.config.bandwidth,
            'noise_dbm': self.config.noise_dbm,
            'sinr_threshold_db': self.config.sinr_threshold_db,
            'distance_scale': self.config.distance_scale,
            'pathloss_a': self.config.pathloss_a,
            'pathloss_b': self.config.pathloss_b,
            'alternating_iterations': self.config.alternating_iterations,
            'random_iterations': self.config.random_iterations,
            'power_optimization_method': self.config.power_optimization_method,
            'weights': self.config.weights,
            'coefficients': self.config.coefficients,
            'seed': self.config.seed + round_idx if self.config.seed is not None else None,
            'mappo_config_path': self.config.mappo_config_path,
            'mappo_model_path': self.config.mappo_model_path,
        }

    def _allocate_resources(self, positions, round_idx, strategy=None):
        """Allocate resources according to strategy (select neighbors) - with time recording."""
        if strategy is None:
            strategy = self.config.allocation_strategy

        wireless_config = self._get_wireless_config(round_idx)
        allocation_start = time.perf_counter()
        neighbor_lists = None

        try:
            if strategy == 'mappo' and MAPPO_AVAILABLE:
                neighbor_lists = mappo_allocator(positions, wireless_config)
            elif strategy == 'alternating' and ALTERNATING_AVAILABLE:
                neighbor_lists = alternating_allocator(positions, wireless_config)
            elif strategy == 'avg' and AVG_AVAILABLE:
                neighbor_lists = avg_allocator(positions, wireless_config)
            elif strategy == 'random' and RANDOM_AVAILABLE:
                neighbor_lists = random_allocator(positions, wireless_config)
            else:
                print(f"Round {round_idx}: Unknown strategy or allocator unavailable '{strategy}', using random allocation")
                neighbor_lists = random_allocator(positions, wireless_config)
        except Exception as e:
            print(f"Round {round_idx}: {strategy} allocation failed: {e}")
            import traceback
            traceback.print_exc()
            neighbor_lists = random_allocator(positions, wireless_config)

        allocation_end = time.perf_counter()
        allocation_time = allocation_end - allocation_start

        # Check timeout
        if self.config.max_allocation_time > 0 and allocation_time > self.config.max_allocation_time:
            print(f"Warning: Round {round_idx} allocation solving time ({allocation_time:.2f}s) exceeds threshold ({self.config.max_allocation_time}s)")

        # Record allocation time
        if self.config.record_allocation_time:
            self.history['allocation_times'].append(allocation_time)
            self.history['allocation_strategy_times'][strategy] += allocation_time
            self.communication_stats['total_allocation_time'] += allocation_time
            self.communication_stats['allocation_time_history'].append(allocation_time)
            self.communication_stats['max_allocation_time'] = max(
                self.communication_stats['max_allocation_time'], allocation_time
            )

        print(f"Round {round_idx}: {strategy} allocation completed, time {allocation_time:.3f}s")
        return neighbor_lists

    def _aggregate_models_weighted(self, terminal_id, neighbor_ids):
        """Weighted model aggregation."""
        if not neighbor_ids:
            return

        terminal = self.terminals[terminal_id]

        neighbor_params = []
        neighbor_weights = []

        for nid in neighbor_ids:
            neighbor = self.terminals[nid]
            neighbor_params.append(neighbor.get_parameters())
            neighbor_weights.append(neighbor.weight)

        total_weight = sum(neighbor_weights)
        normalized_weights = [w / total_weight for w in neighbor_weights]

        current_params = terminal.get_parameters()
        aggregated_params = []
        for param_idx in range(len(current_params)):
            weighted_sum = torch.zeros_like(current_params[param_idx])
            for w, nparam in zip(normalized_weights, neighbor_params):
                weighted_sum += w * nparam[param_idx]
            alpha = self.config.mixing_factor
            aggregated_params.append(alpha * current_params[param_idx] + (1 - alpha) * weighted_sum)

        terminal.set_parameters(aggregated_params)

    def evaluate(self):
        """Evaluation: each terminal is evaluated on the global test set, then weighted average is computed."""
        terminal_accuracies = []
        terminal_losses = []

        for terminal in self.terminals:
            acc, loss = self._evaluate_single_terminal(terminal, self.test_loader)
            terminal_accuracies.append(acc)
            terminal_losses.append(loss)

        weighted_accuracy = np.average(terminal_accuracies, weights=self.config.weights)
        simple_accuracy = np.mean(terminal_accuracies)

        return weighted_accuracy, simple_accuracy, terminal_accuracies, terminal_losses

    def _evaluate_single_terminal(self, terminal, dataloader):
        """Evaluate a single terminal."""
        terminal.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = terminal.model(data)
                loss = criterion(output, target)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                total_loss += loss.item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss

    def _process_round(self, round_idx, neighbor_lists):
        """Process a single round of local training, aggregation, and evaluation (common part)."""
        # Local training
        local_start = time.time()
        for terminal in self.terminals:
            terminal.local_train(self.config.local_epochs)
        local_time = time.time() - local_start
        self.training_times['local_training_times'].append(local_time)

        # Aggregation
        agg_start = time.time()
        for tid in range(self.config.num_agents):
            self._aggregate_models_weighted(tid, neighbor_lists[tid])
        agg_time = time.time() - agg_start
        self.training_times['aggregation_times'].append(agg_time)

        # Evaluation
        if round_idx % self.config.eval_frequency == 0:
            eval_start = time.time()
            w_acc, s_acc, term_accs, term_losses = self.evaluate()
            eval_time = time.time() - eval_start
            self.training_times['evaluation_times'].append(eval_time)

            avg_neighbors = np.mean([len(n) for n in neighbor_lists])
            total_links = sum(len(n) for n in neighbor_lists)
            max_possible = self.config.num_agents * (self.config.num_agents - 1)
            connectivity_ratio = total_links / max_possible if max_possible > 0 else 0

            self.history['communication_rounds'].append(round_idx)
            self.history['weighted_accuracy'].append(w_acc)
            self.history['simple_accuracy'].append(s_acc)
            self.history['terminal_accuracies'].append(term_accs)
            self.history['terminal_losses'].append(term_losses)
            self.history['neighbor_counts'].append(avg_neighbors)
            self.history['connectivity_stats'].append({
                'total_links': total_links, 'max_possible_links': max_possible,
                'connectivity_ratio': connectivity_ratio, 'avg_neighbors': avg_neighbors
            })

            alloc_time = self.history['allocation_times'][-1] if self.config.record_allocation_time and self.history['allocation_times'] else 0
            print(f"Round {round_idx:3d}/{self.config.communication_rounds}: "
                  f"Weighted Acc = {w_acc:.2f}%, Simple Avg = {s_acc:.2f}%, Avg Neighbors = {avg_neighbors:.2f}, Connectivity Ratio = {connectivity_ratio:.1%}")
            if self.config.record_allocation_time:
                print(f"      Time stats: Allocation = {alloc_time:.3f}s, Local Training = {local_time:.3f}s, Aggregation = {agg_time:.3f}s, Evaluation = {eval_time:.3f}s")

            if round_idx % 10 == 0:
                print("     Terminal accuracies:")
                for i, acc in enumerate(term_accs):
                    print(f"      Terminal {i}: {acc:.2f}% (weight = {self.terminals[i].weight:.2f})")

    def _train_sequential(self):
        """Sequential training."""
        print(f"Using sequential training mode, allocation strategy: {self.config.allocation_strategy}")
        total_start = time.time()

        for round_idx in range(self.config.communication_rounds):
            round_start = time.time()
            positions = self.trajectory_generator.get_positions(round_idx)
            neighbor_lists = self._allocate_resources(positions, round_idx)
            self._process_round(round_idx, neighbor_lists)
            self.training_times['round_times'].append(time.time() - round_start)

            if round_idx % self.config.save_frequency == 0:
                self.save_checkpoint(round_idx)

        self.training_times['total_training_time'] = time.time() - total_start

    def _train_threaded(self):
        """Multi-threaded training."""
        print(f"Using multi-threaded training mode, allocation strategy: {self.config.allocation_strategy}")
        total_start = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            for round_idx in range(self.config.communication_rounds):
                round_start = time.time()
                positions = self.trajectory_generator.get_positions(round_idx)
                neighbor_lists = self._allocate_resources(positions, round_idx)

                # Parallel local training
                futures = [executor.submit(t.local_train, self.config.local_epochs) for t in self.terminals]
                for f in futures:
                    f.result()

                # Sequential aggregation (needs parameter update)
                agg_start = time.time()
                for tid in range(self.config.num_agents):
                    self._aggregate_models_weighted(tid, neighbor_lists[tid])
                agg_time = time.time() - agg_start
                self.training_times['aggregation_times'].append(agg_time)

                # Evaluation
                if round_idx % self.config.eval_frequency == 0:
                    eval_start = time.time()
                    w_acc, s_acc, term_accs, term_losses = self.evaluate()
                    eval_time = time.time() - eval_start
                    self.training_times['evaluation_times'].append(eval_time)

                    avg_neighbors = np.mean([len(n) for n in neighbor_lists])
                    total_links = sum(len(n) for n in neighbor_lists)
                    max_possible = self.config.num_agents * (self.config.num_agents - 1)
                    connectivity_ratio = total_links / max_possible if max_possible > 0 else 0

                    self.history['communication_rounds'].append(round_idx)
                    self.history['weighted_accuracy'].append(w_acc)
                    self.history['simple_accuracy'].append(s_acc)
                    self.history['terminal_accuracies'].append(term_accs)
                    self.history['terminal_losses'].append(term_losses)
                    self.history['neighbor_counts'].append(avg_neighbors)
                    self.history['connectivity_stats'].append({
                        'total_links': total_links, 'max_possible_links': max_possible,
                        'connectivity_ratio': connectivity_ratio, 'avg_neighbors': avg_neighbors
                    })

                    alloc_time = self.history['allocation_times'][-1] if self.config.record_allocation_time and self.history['allocation_times'] else 0
                    print(f"Round {round_idx:3d}/{self.config.communication_rounds}: "
                          f"Weighted Acc = {w_acc:.2f}%, Simple Avg = {s_acc:.2f}%, Avg Neighbors = {avg_neighbors:.2f}, Connectivity Ratio = {connectivity_ratio:.1%}")
                    if self.config.record_allocation_time:
                        print(f"      Allocation solving time: {alloc_time:.3f}s")

                    if round_idx % 10 == 0:
                        print("     Terminal accuracies:")
                        for i, acc in enumerate(term_accs):
                            print(f"      Terminal {i}: {acc:.2f}% (weight = {self.terminals[i].weight:.2f})")

                self.training_times['round_times'].append(time.time() - round_start)

                if round_idx % self.config.save_frequency == 0:
                    self.save_checkpoint(round_idx)

        self.training_times['total_training_time'] = time.time() - total_start

    def train(self):
        """Start training."""
        print(f"\nStarting DFL training")
        print(f"Allocation strategy: {self.config.allocation_strategy}")
        print(f"Communication rounds: {self.config.communication_rounds}")
        print(f"Local training epochs: {self.config.local_epochs}")
        print(f"Terminal weights: {self.config.weights}")

        # Initial evaluation
        print("\nInitial evaluation...")
        w_acc, s_acc, term_accs, _ = self.evaluate()
        print(f"Initial weighted accuracy: {w_acc:.2f}%")
        print(f"Initial simple average: {s_acc:.2f}%")
        for i, acc in enumerate(term_accs):
            print(f"  Terminal {i}: {acc:.2f}% (weight = {self.terminals[i].weight:.2f})")

        # Choose training mode
        if self.config.communication_mode == 'sequential':
            self._train_sequential()
        elif self.config.communication_mode == 'thread':
            self._train_threaded()
        else:
            print(f"Unknown communication mode {self.config.communication_mode}, using sequential mode")
            self._train_sequential()

        # Final evaluation
        print("\nFinal evaluation...")
        w_acc, s_acc, term_accs, _ = self.evaluate()
        print(f"Final weighted accuracy: {w_acc:.2f}%")
        print(f"Final simple average: {s_acc:.2f}%")
        for i, acc in enumerate(term_accs):
            print(f"  Terminal {i}: {acc:.2f}% (weight = {self.terminals[i].weight:.2f})")

        total_time = self.training_times['total_training_time']
        print(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

        self.save_checkpoint(self.config.communication_rounds - 1)
        results_path = self.save_results()
        plot_path = plot_results(self.history, self.config, self.config.result_dir, self.config.allocation_strategy)
        if self.config.record_allocation_time:
            plot_time_breakdown(self.history, self.config, self.config.result_dir, self.config.allocation_strategy)

        return self.history, results_path, plot_path

    def save_checkpoint(self, round_idx):
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.config.model_dir, f"checkpoint_round_{round_idx}.pt")
        checkpoint = {
            'round': round_idx,
            'terminals': [],
            'history': self.history,
            'config': self.config.__dict__,
            'communication_stats': self.communication_stats,
            'training_times': self.training_times
        }
        for terminal in self.terminals:
            checkpoint['terminals'].append({
                'id': terminal.id,
                'model_state': terminal.get_model_state(),
                'optimizer_state': terminal.optimizer.state_dict(),
                'weight': terminal.weight,
                'train_loss_history': terminal.train_loss_history,
                'train_acc_history': terminal.train_acc_history
            })
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    def save_results(self):
        """Save training results."""
        results = {
            'config': self.config.__dict__,
            'history': self.history,
            'communication_stats': self.communication_stats,
            'training_times': self.training_times,
            'terminal_weights': self.config.weights,
            'allocation_strategy': self.config.allocation_strategy,
            'allocation_time_stats': {
                'total': self.communication_stats.get('total_allocation_time', 0),
                'average': self.communication_stats.get('avg_allocation_time', 0),
                'maximum': self.communication_stats.get('max_allocation_time', 0),
                'history': self.communication_stats.get('allocation_time_history', []),
                'strategy_times': self.history.get('allocation_strategy_times', {})
            }
        }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config.result_dir, f"dfl_results_{self.config.allocation_strategy}_{timestamp}.json")

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(results_path, 'w') as f:
            json.dump(convert(results), f, indent=2)

        print(f"Results saved to: {results_path}")
        return results_path