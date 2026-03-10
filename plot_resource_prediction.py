import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman globally
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# raw data provided by user with header for clarity:
# ID | Category | TCAM (%) | SRAM (%) | Stage | Depth | Diff
raw_data = """
29	Actual	31.94	0.42	4	35	0.2064
	Predicted	31.7336	0.4001	4		
28	Actual	31.94	0.42	4	38	0.2256
	Predicted	31.7144	0.4001	4		
25	Actual	46.88	0.62	6	17	0.9054
	Predicted	47.7854	0.6490	6		
15	Actual	55.21	0.73	7	20	0.2248
	Predicted	54.9852	0.6780	7		
8	Actual	55.21	0.73	7	30	0.2993
	Predicted	54.9107	0.7190	7		
18	Actual	54.17	0.73	7	19	0.4243
	Predicted	54.5943	0.7326	7		
23	Actual	51.04	0.73	7	18	0.5036
	Predicted	51.5436	0.7105	7		
4	Actual	55.56	0.73	7	33	0.6844
	Predicted	54.8756	0.7348	7		
19	Actual	54.17	0.73	7	15	0.7197
	Predicted	54.8897	0.7442	7		
26	Actual	52.08	0.73	7	11	0.8057
	Predicted	52.8857	0.7511	7		
21	Actual	53.12	0.73	7	14	1.0558
	Predicted	54.1758	0.7348	7		
9	Actual	53.12	0.73	7	36	1.1092
	Predicted	54.2292	0.7149	7		
10	Actual	53.12	0.73	7	37	1.1092
	Predicted	54.2292	0.7149	7		
24	Actual	51.04	0.73	7	13	1.2006
	Predicted	52.2406	0.7334	7		
16	Actual	56.25	0.73	7	25	1.3314
	Predicted	54.9186	0.7098	7		
27	Actual	53.12	0.73	7	10	1.3953
	Predicted	54.5153	0.7149	7		
20	Actual	51.04	0.73	7	26	1.4068
	Predicted	52.4468	0.7185	7		
17	Actual	52.08	0.73	7	27	1.4786
	Predicted	53.5586	0.7149	7		
"""

def parse_data(data_str):
    # Convert '实际消耗' and '预测消耗' if they appear in text (safeguard)
    data_str = data_str.replace("实际消耗", "Actual").replace("预测消耗", "Predicted")
    lines = [l.strip() for l in data_str.strip().split('\n') if l.strip()]
    records = []
    
    for i in range(0, len(lines), 2):
        act_parts = lines[i].split()
        pred_parts = lines[i+1].split()
        
        try:
            # According to header: ID, Cat, TCAM, SRAM, Stage, Depth, Diff
            depth = float(act_parts[5])
            
            # Resource columns: TCAM(2), SRAM(3), Stage(4)
            r1_act, r1_pred = float(act_parts[2]), float(pred_parts[1]) # TCAM
            r2_act, r2_pred = float(act_parts[3]), float(pred_parts[2]) # SRAM
            r3_act, r3_pred = float(act_parts[4]), float(pred_parts[3]) # Stage
            
            records.append({
                'depth': depth,
                'resources': [
                    {'name': 'TCAM Usage (%)', 'act': r1_act, 'pred': r1_pred},
                    {'name': 'SRAM Usage (%)', 'act': r2_act, 'pred': r2_pred},
                    {'name': 'Stages (Count)', 'act': r3_act, 'pred': r3_pred}
                ]
            })
        except (IndexError, ValueError) as e:
            continue
            
    # Sort by Depth for clear line plotting
    records.sort(key=lambda x: x['depth'])
    return records

def plot_resources(records):
    depth_vals = [r['depth'] for r in records]
    num_resources = 3 # Only TCAM, SRAM, Stage
    
    fig, axes = plt.subplots(num_resources, 1, figsize=(10, 11), sharex=True)
    
    # Colors and styles
    color_act = '#2980b9'  # Firm Blue
    color_pred = '#c0392b' # Solid Red
    
    for i in range(num_resources):
        ax = axes[i]
        res_name = records[0]['resources'][i]['name']
        
        act_vals = [r['resources'][i]['act'] for r in records]
        pred_vals = [r['resources'][i]['pred'] for r in records]
        
        # Plotting
        ax.plot(depth_vals, act_vals, marker='o', markersize=5, label='Actual', 
                color=color_act, linewidth=1.5)
        ax.plot(depth_vals, pred_vals, marker='x', markersize=6, label='Predicted', 
                color=color_pred, linestyle='--', linewidth=1.5)
        
        # Axes labels and title (Times New Roman set via global rc)
        ax.set_title(f'Analysis of {res_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Resource Consumption', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=10)
        
    axes[-1].set_xlabel('Depth', fontsize=12)
    
    plt.tight_layout()
    output_filename = 'resource_consumption_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_filename}")

if __name__ == "__main__":
    data_records = parse_data(raw_data)
    if data_records:
        plot_resources(data_records)
    else:
        print("Failed to parse data.")

if __name__ == "__main__":
    data_records = parse_data(raw_data)
    if data_records:
        plot_resources(data_records)
    else:
        print("No valid data parsed.")
