def get_dfl_config(parser):
    """Add DFL environment specific parameters to the parser."""
    
    # Basic DFL parameters
    parser.add_argument("--num_agents", type=int, default=5, 
                       help="number of agents (terminals) in DFL")
    parser.add_argument("--num_channels", type=int, default=8, 
                       help="number of available frequency channels")
    parser.add_argument("--bandwidth", type=float, default=1e6, 
                       help="channel bandwidth in Hz")
    parser.add_argument("--noise_dbm", type=float, default=-174, 
                       help="noise power spectral density in dBm/Hz")
    parser.add_argument("--gamma_th", type=float, default=10.0, 
                       help="SINR threshold in dB")
    parser.add_argument("--max_power", type=float, default=0.2, 
                       help="maximum transmit power per terminal in Watt")
    
    
    # Normalization method
    parser.add_argument("--normalization_method", type=str, default="log_standard",
                       choices=['none', 'log_standard', 'log_minmax', 'relative', 'simple'],
                       help="channel gain normalization method")
    
    # Penalty coefficients
    parser.add_argument("--channel_penalty_weight", type=float, default=0.2,
                       help="penalty weight for channel repetition")
    parser.add_argument("--bidirectional_same_channel_penalty_weight", type=float, default=0.2,
                       help="penalty weight for bidirectional same channel")
    parser.add_argument("--aux_reward_weight", type=float, default=0.1,
                       help="reward weight for successful links")
    
    # Reward scaling factor
    parser.add_argument("--reward_scaling_factor", type=float, default=10.0,
                       help="reward scaling factor for training stability")
    
    # Deployment radius
    parser.add_argument("--deployment_radius", type=float, default=200.0,
                       help="deployment radius in meters")

    
    return parser