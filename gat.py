import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GuidedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, concat=True,
                 dropout=0.2, feature_guidance_weight=0, **kwargs):
        super(GuidedGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.feature_guidance_weight = feature_guidance_weight

        # Initialize learnable parameters
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if concat:
            self.output_dim = heads * out_channels
        else:
            self.output_dim = out_channels

        self.reset_parameters()
        self.attention_weights = None

    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index):
        """
        Algorithm: Reward-Guided Graph Attention
        Input: 
            x: Node features
            edge_index: Graph connectivity
        Process:
        1. Extract reward values from last dimension of node features
        2. Calculate reward-guided weights:
           - For nodes with smaller rewards, assign higher weights
           - Normalize weights using L1 normalization
        3. Transform node features through linear layer
        4. Reshape features for multi-head attention
        5. Propagate messages through graph structure
        """
        rewards = x[:, -1:]
        reward_weights = 1.0 / (torch.abs(rewards) + 1e-6)
        reward_weights = F.normalize(reward_weights, p=1, dim=0)
        
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)
        
        return self.propagate(edge_index, x=x,
                            reward_weights=reward_weights,
                            size=None)

    def message(self, edge_index_i, x_i, x_j, reward_weights_j, size_i):
        """
        Algorithm: Attention-based Message Passing
        Input:
            edge_index_i: Target nodes of edges
            x_i: Features of target nodes
            x_j: Features of source nodes
            reward_weights_j: Reward-based weights of source nodes
            size_i: Number of target nodes
        Process:
        1. Calculate raw attention scores:
           alpha = (x_i * att_l).sum(-1) + (x_j * att_r).sum(-1)
        2. Apply non-linearity using LeakyReLU
        3. Incorporate reward guidance:
           guided_alpha = alpha + feature_guidance_weight * log(reward_weights)
        4. Normalize attention scores using softmax
        5. Apply dropout during training
        6. Weight source node features with attention scores
        Output:
            Updated node features incorporating attention and rewards
        """
        # Pseudo-implementation
        alpha = self._compute_attention_scores(x_i, x_j)
        guided_alpha = self._apply_reward_guidance(alpha, reward_weights_j)
        alpha = self._normalize_and_dropout(guided_alpha, edge_index_i, size_i)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        """
        Algorithm: Feature Aggregation
        Input: 
            aggr_out: Aggregated features from all neighbors
        Process:
        1. If using concatenation:
           - Reshape output to combine all attention heads
        2. If not using concatenation:
           - Average across attention heads
        Output:
            Final node representations
        """
        if self.concat:
            return aggr_out.view(-1, self.heads * self.out_channels)
        else:
            return aggr_out.mean(dim=1)


class GuidedGATRegression(torch.nn.Module):
    """
    Algorithm: Guided Graph Attention Network for Regression
    Architecture:
    1. Multi-head attention layers with reward guidance
    2. Non-linear activation and dropout for regularization
    3. Final projection layer for output prediction
    
    Key Features:
    - Utilizes reward signals to guide attention
    - Multiple attention heads for diverse feature learning
    - Residual connections for stable training
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.2, feature_guidance_weight=0):
        super(GuidedGATRegression, self).__init__()
        # Pseudo-implementation
        self._init_network_layers(input_dim, hidden_dim, output_dim, heads, dropout)
        self.feature_guidance_weight = feature_guidance_weight

    def forward(self, x, edge_index):
        """
        Forward pass algorithm:
        1. First attention layer:
           - Apply guided attention mechanism
           - Non-linear activation and dropout
        2. Second attention layer:
           - Aggregate multi-head features
           - Apply final attention mechanism
        3. Output projection:
           - Transform to target dimension
        """
        # Pseudo-implementation
        x = self._first_attention_layer(x, edge_index)
        x = self._second_attention_layer(x, edge_index)
        return self._output_projection(x)

    def get_attention_weights(self):
        """Get attention weights from both layers"""
        return {
            'layer1': self.conv1.attention_weights,
            'layer2': self.conv2.attention_weights
        }


def generate_random_graph_data(num_nodes=5, num_features=4, output_dim=3, avg_degree=4):
    """
    Generate random graph data for regression task with output_dim dimensional target
    """
    # Generate node features (intentionally making some nodes have small feature values)
    x = torch.randn(num_nodes, num_features)
    small_features_mask = torch.randint(0, 2, (num_nodes,)).bool()
    x[small_features_mask] *= 0.1

    # Generate random edges
    num_edges = int(num_nodes * avg_degree)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Generate target vectors using a simple non-linear function
    y = torch.sin(x[:, :output_dim] * 0.5) + torch.cos(x[:, :output_dim] * 0.3)
    y = y + torch.randn_like(y) * 0.1  # Add some noise

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def train(model, data, optimizer):
    """
    Algorithm: Training Procedure
    Process:
    1. Forward pass through model
    2. Calculate loss using MSE
    3. Backpropagate gradients
    4. Update model parameters
    
    Hyperparameters:
    - Learning rate
    - Weight decay
    - Dropout rate
    - Number of attention heads
    """
    # Pseudo-implementation
    model.train()
    loss = _compute_and_update(model, data, optimizer)
    return loss


@torch.no_grad()
def evaluate(model, data, mask=None):
    """
    Algorithm: Model Evaluation
    Process:
    1. Forward pass without gradient computation
    2. Calculate MSE and MAE metrics
    3. Apply mask for specific node evaluation
    """
    # Pseudo-implementation
    model.eval()
    metrics = _compute_metrics(model, data, mask)
    return metrics


def analyze_attention(model, edge_index, epoch, num_nodes):
    """
    Algorithm: Attention Analysis
    Process:
    1. Extract attention weights
    2. Calculate node importance scores
    3. Identify top-k influential nodes
    4. Analyze node connectivity patterns
    """
    # Pseudo-implementation
    attention_scores = _extract_attention_weights(model)
    node_importance = _calculate_node_importance(attention_scores, edge_index)
    _analyze_top_nodes(node_importance, num_nodes)


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(12345)
    np.random.seed(12345)

    # Set output vector dimension
    output_dim = 3

    # Generate data
    data = generate_random_graph_data(output_dim=output_dim)

    # Split into training and test sets
    num_nodes = data.x.size(0)
    node_indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(node_indices, test_size=0.2, random_state=42)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Create model
    model = GuidedGATRegression(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=output_dim,
        heads=4,
        dropout=0,  # Lower dropout for regression tasks
        feature_guidance_weight = 10
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Training loop
    epochs = 200
    train_losses = []
    train_mses = []
    test_mses = []

    print(f"Starting training... Target is to predict {output_dim}-dimensional vector")
    for epoch in range(epochs):
        # Train
        loss = train(model, data, optimizer)

        # Evaluate
        train_mse, train_mae = evaluate(model, data, train_mask)
        test_mse, test_mae = evaluate(model, data, test_mask)

        # Record metrics
        train_losses.append(loss)
        train_mses.append(train_mse)
        test_mses.append(test_mse)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, '
                  f'Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, '
                  f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')
        analyze_attention(model, data.edge_index, epoch + 1, num_nodes)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')

    plt.subplot(1, 2, 2)
    plt.plot(train_mses, label='Train MSE')
    plt.plot(test_mses, label='Test MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Output prediction samples for comparison
    with torch.no_grad():
        model.eval()
        pred = model(data.x, data.edge_index)
        print("\nPrediction samples comparison (first 5 test nodes):")
        test_indices = test_idx[:5]
        print("Ground truth:")
        print(data.y[test_indices].numpy())
        print("Predictions:")
        print(pred[test_indices].numpy())


if __name__ == '__main__':
    main()