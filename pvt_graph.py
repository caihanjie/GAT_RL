import torch
import numpy as np
import os
import shutil

class PVTGraph:
    def __init__(self):
        # PVT corner definitions
        self.pvt_corners = {
            'tt_027C_1v80': [1, 0, 0, 0, 0, 27 ,1.8],  
            'ff_027C_1v80': [0, 1, 0, 0, 0, 27 ,1.8],
            'ss_027C_1v80': [0, 0, 1, 0, 0, 27 ,1.8],
            'fs_027C_1v80': [0, 0, 0, 1, 0, 27 ,1.8],
            'sf_027C_1v80': [0, 0, 0, 0, 1, 27 ,1.8]

            # 'tt_027C_1v80': [1, 0, 0, 0, 0, 27 ,1.8],  

            # 'fs_-25C_1v62': [0, 1, 0, 0, 0, -25 ,1.62],
            # 'fs_-25C_1v98': [0, 1, 0, 0, 0, -25 ,1.98],
            # 'fs_125C_1v80': [0, 1, 0, 0, 0, 125 ,1.62],
            # 'fs_125C_1v98': [0, 1, 0, 0, 0, 125 ,1.98],

            # 'sf_-25C_1v62': [0, 0, 1, 0, 0, -25 ,1.62],
            # 'sf_-25C_1v98': [0, 0, 1, 0, 0, -25 ,1.98],
            # 'sf_125C_1v80': [0, 0, 1, 0, 0, 125 ,1.62],
            # 'sf_125C_1v98': [0, 0, 1, 0, 0, 125 ,1.98],

            # 'ff_-25C_1v62': [0, 0, 0, 1, 0, -25 ,1.62],
            # 'ff_-25C_1v98': [0, 0, 0, 1, 0, -25 ,1.98],
            # 'ff_125C_1v80': [0, 0, 0, 1, 0, 125 ,1.62],
            # 'ff_125C_1v98': [0, 0, 0, 1, 0, 125 ,1.98],

            # 'ss_-25C_1v62': [0, 0, 0, 0, 1, -25 ,1.62],
            # 'ss_-25C_1v98': [0, 0, 0, 0, 1, -25 ,1.98],
            # 'ss_125C_1v80': [0, 0, 0, 0, 1, 125 ,1.62],
            # 'ss_125C_1v98': [0, 0, 0, 0, 1, 125 ,1.98]
        }
        
        self.PWD = os.getcwd()
        self.SPICE_NETLIST_DIR = f'{self.PWD}/simulations'
        
        
        
        # Initialize PVT graph node features
        self.num_corners = len(self.pvt_corners)
        self.corner_dim = 22
        self.node_features = np.zeros((self.num_corners, self.corner_dim))  
        
        # Initialize features for each corner
        for i, (corner, pvt_code) in enumerate(self.pvt_corners.items()):
            self.node_features[i, :7] = pvt_code  # PVT encoding
            # Initialize performance metrics and reward to worst values
            self.node_features[i, 7:21] = -np.inf  # Performance metrics
            self.node_features[i, 21] = -np.inf    # Reward
            
        # Define graph edges - complete graph
        edges = []
        for i in range(self.num_corners):
            for j in range(self.num_corners):
                if i != j:
                    edges.append([i, j])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        


    def _clean_pvt_dirs(self):
        """Clean existing PVT corner directories"""
        # Define directory prefixes to clean
        corner_prefixes = ['ss', 'ff', 'tt', 'sf', 'fs']
        
        # Iterate through all folders in the specified directory
        for corner in os.listdir(self.SPICE_NETLIST_DIR):
            # Get full directory path
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            if os.path.isdir(corner_dir) and any(corner.startswith(prefix) for prefix in corner_prefixes):
                print(f"Removing existing corner directory: {corner_dir}")
                shutil.rmtree(corner_dir)

    def _create_pvt_dirs(self):
        """Create directories for each PVT corner"""
        for corner in self.pvt_corners.keys():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            os.makedirs(corner_dir)
            
            # Create .spiceinit file
            spiceinit_content ="""* ngspice initialization for sky130
* assert BSIM compatibility mode with "nf" vs. "W"
set ngbehavior=hsa
* "nomodcheck" speeds up loading time
set ng_nomodcheck
set num_threads=8"""
            
            spiceinit_path = os.path.join(corner_dir, '.spiceinit')
            with open(spiceinit_path, 'w') as f:
                f.write(spiceinit_content)
                
    def _create_pvt_netlists(self):
        """Create simulation files for each PVT corner"""
        # Read original netlist file
        with open(f'{self.SPICE_NETLIST_DIR}/AMP_NMCF_ACDC.cir', 'r') as f:
            netlist_content = f.readlines()
            
        # Create netlist for each corner
        for corner, params in self.pvt_corners.items():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            # Copy necessary files
            self._copy_support_files(corner_dir)
            
            # Modify netlist content
            corner_netlist = []

            # Parse corner name to get process corner
            process = corner.split('_')[0]  # tt/ff/ss/fs/sf
            
            # Iterate through original netlist content and modify based on PVT parameters
            for line in netlist_content:
                if line.startswith('.temp'):
                    # Modify temperature
                    corner_netlist.append(f'.temp {params[5]}\n')
                elif line.startswith('.include') and 'tt.spice' in line:
                    # Modify process corner path
                    corner_netlist.append(f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/{process}.spice\n')
                elif line.startswith('.PARAM supply_voltage'):
                    # Modify supply voltage
                    corner_netlist.append(f'.PARAM supply_voltage = {params[6]}\n')
                else:
                    corner_netlist.append(line)
            corner_netlist.insert(1, f'* PVT Corner: {corner}\n')
            # Save modified netlist
            netlist_path = os.path.join(corner_dir, f'AMP_NMCF_ACDC_{corner}.cir')
            with open(netlist_path, 'w') as f:
                f.writelines(corner_netlist)
    
    def _create_pvt_netlists_tran(self):
        """Create transient simulation files for each PVT corner"""
        # Read original netlist file
        with open(f'{self.SPICE_NETLIST_DIR}/AMP_NMCF_Tran.cir', 'r') as f:
            netlist_content = f.readlines()
            
        # Create netlist for each corner
        for corner, params in self.pvt_corners.items():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            # Copy necessary files
            self._copy_support_files(corner_dir)
            
            # Modify netlist content
            corner_netlist = []

            # Parse corner name to get process corner
            process = corner.split('_')[0]  # tt/ff/ss/fs/sf
            
            # Iterate through original netlist content and modify based on PVT parameters
            for line in netlist_content:
                if line.startswith('.temp'):
                    # Modify temperature
                    corner_netlist.append(f'.temp {params[5]}\n')
                elif line.startswith('.include') and 'tt.spice' in line:
                    # Modify process corner path
                    corner_netlist.append(f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/{process}.spice\n')
                elif line.startswith('.PARAM supply_voltage'):
                    # Modify supply voltage
                    corner_netlist.append(f'.PARAM supply_voltage = {params[6]}\n')
                else:
                    corner_netlist.append(line)
            corner_netlist.insert(1, f'* PVT Corner: {corner}\n')
            # Save modified netlist
            netlist_path = os.path.join(corner_dir, f'AMP_NMCF_Tran_{corner}.cir')
            with open(netlist_path, 'w') as f:
                f.writelines(corner_netlist)

    def _copy_support_files(self, corner_dir):
        """Copy support files to corner directory"""
        support_files = [
            # Add support files as needed
        ]
        
        for file in support_files:
            src = os.path.join(self.SPICE_NETLIST_DIR, file)
            dst = os.path.join(corner_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
    def get_corner_netlist_path(self, corner_idx):
        """Get netlist path for specified corner"""
        corner_name = list(self.pvt_corners.keys())[corner_idx]
        return os.path.join(self.SPICE_NETLIST_DIR, corner_name, f'AMP_NMCF_ACDC_{corner_name}.cir')

    def update_performance_and_reward(self, corner_idx, new_performance, new_reward):
        """
        Update best performance and reward for specified corner
        
        Args:
            corner_idx: Index of PVT corner
            new_performance: New performance metrics list [phase_margin, dcgain, PSRP, ...]
            new_reward: New reward value
        """
        current_reward = self.node_features[corner_idx, 21]
        # Only update performance and reward when new reward is better
        if new_reward > current_reward:
            performance_array = np.array(list(new_performance.values()), dtype=np.float32)
            self.node_features[corner_idx, 7:21] = performance_array  # Update performance metrics
            self.node_features[corner_idx, 21] = new_reward           # Update reward

    def update_performance_and_reward_r(self, corner_idx, info_dict, reward):
        """
        Force update performance and reward for specified corner
        
        Args:
            corner_idx: Index of PVT corner
            info_dict: Performance metrics dictionary {'phase_margin': float, 'dcgain': float, ...}
            reward: New reward value
        """
        # Extract performance metrics list from info dictionary
        performance_array = np.array(list(info_dict.values()), dtype=np.float32)
        self.node_features[corner_idx, 7:21] = performance_array  # Update performance metrics
        self.node_features[corner_idx, 21] = reward               # Update reward

    def get_corner_name(self, idx):
        """Get corner name by index"""
        return list(self.pvt_corners.keys())[idx]
    
    def get_corner_idx(self, corner_name):
        """Get index by corner name"""
        return list(self.pvt_corners.keys()).index(corner_name)

    def get_best_corner(self):
        """
        Get corner with highest current reward
        
        Returns:
            Tuple (corner_idx, best_reward)
        """
        rewards = self.node_features[:, 21]
        best_idx = np.argmax(rewards)
        return best_idx, rewards[best_idx]

    def get_graph_features(self):
        """
        Get feature representation of PVT graph
        
        Returns:
            Node feature matrix (torch.Tensor)
        """
        return torch.tensor(self.node_features, dtype=torch.float32).to(self.device) 