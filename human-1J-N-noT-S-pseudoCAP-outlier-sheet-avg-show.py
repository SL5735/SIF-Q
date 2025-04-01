import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.spatial import Delaunay
from scipy import stats
import numpy as np
from PIL import Image

# ================= 参数设置 =================
color = "blue"
m = 2  ###########超参数1
version = f"{color}-noT-S-pseudoCAP-m{m}-de3.0-avg_show"  


R_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150]  # 超参数2

A = np.array([95, 60, 20, 4, 0])
T_values = np.arange(1.0, 1.3, 0.1)  # T 从 1.0 到 1.2，步长 0.1


file = 'human-1J-N'
tree = f'treecut-unique' 
count = f"{file}_{color}_{tree}_T_E"  

# ================= 路径设置 =================

base_path = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\human\1J-N\human_dpienhanced'
deal_path = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\human\1J-N'


output_dir = os.path.join(deal_path, f"try_{version}")
os.makedirs(output_dir, exist_ok=True)

# 
tree_dir = os.path.join(deal_path, f"{color}_{tree}")
show_dir = os.path.join(output_dir, count)
fit_dir = os.path.join(output_dir, f"fit{count}")

for path in [tree_dir, show_dir, fit_dir]:
    os.makedirs(path, exist_ok=True)

# 
show_tree_dir = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\human\1J-N\show-tree'
os.makedirs(show_tree_dir, exist_ok=True)

###############################################
# =================读取 B 图并保存统计信息 =================
excel_B_stats_path = os.path.join(deal_path, f'B_stats_{color}_{tree}.xlsx')
if not os.path.exists(excel_B_stats_path):   # 
    with pd.ExcelWriter(excel_B_stats_path, engine='xlsxwriter') as writer:
        for F in A:
          
            image_b_path = os.path.join(deal_path, f'{F}%_c2.tif')
            try:
                image_b = Image.open(image_b_path).convert("L")
                image_b_array = np.array(image_b, dtype=np.float32)
            except Exception as e:
                print(f"⚠ B 图加载失败: {image_b_path}，错误信息: {e}")
                continue

            # 
            positive_values = image_b_array[image_b_array > 0]
            if positive_values.size == 0:
                stats_dict = {
                    "最小值": None,
                    "最大值": None,
                    "均值": None,
                    "中位数": None,
                    "众数": None,
                    "25%分位数": None,
                    "50%分位数": None,
                    "75%分位数": None,
                    "95%分位数": None,
                    "95-100%区间": None
                }
            else:
                min_val = np.min(positive_values)
                max_val = np.max(positive_values)
                mean_val = np.mean(positive_values)
                median_val = np.median(positive_values)
                mode_result = stats.mode(positive_values, nan_policy='omit')
                mode_val = np.atleast_1d(mode_result.mode)[0] if np.atleast_1d(mode_result.count)[0] > 0 else None
                percentiles = np.percentile(positive_values, [25, 50, 75, 95])
                range_95_100 = f"{percentiles[3]}-{max_val}"
                stats_dict = {
                    "最小值": min_val,
                    "最大值": max_val,
                    "均值": mean_val,
                    "中位数": median_val,
                    "众数": mode_val,
                    "25%分位数": percentiles[0],
                    "50%分位数": percentiles[1],
                    "75%分位数": percentiles[2],
                    "95%分位数": percentiles[3],
                    "95-100%区间": range_95_100
                }
            df_stats = pd.DataFrame([stats_dict])
            sheet_name = f'F_{F}'
            df_stats.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"B 图统计数据已保存到 Sheet: {sheet_name}")
    print(f"所有 B 图统计数据已保存到 {excel_B_stats_path}")
else:
    print(f"检测到已存在的 B 图统计文件，跳过统计处理: {excel_B_stats_path}")

# =================Excel文件 =================

excel_show_all_path = os.path.join(show_dir, f'analysis_results_{count}.xlsx')

with pd.ExcelWriter(excel_show_all_path, engine='openpyxl') as writer_all:

    # ##########===== 外层循环=====
    for R in R_values:
        

        excel_tree_path = os.path.join(tree_dir, f'edge_analysis_{color}_{R}.xlsx')
        # 
        if os.path.exists(excel_tree_path):
            print(f"检测到已存在的第一部分结果文件，跳过R={R}的第一部分处理: {excel_tree_path}")
        else:
         
            with pd.ExcelWriter(excel_tree_path, engine='xlsxwriter') as writer:
                for F in A:
                    print(f"Processing F={F} ...")
                    
                    # 读取 C 图
                    image_c_path = os.path.join(base_path, f'T3-ZLQ {F}% dpi_masks.png')
                    imgC = cv2.imread(image_c_path, cv2.IMREAD_GRAYSCALE)
                    if imgC is None:
                        print(f"⚠ C 图加载失败: {image_c_path}")
                        continue

                 
                    _, binaryC = cv2.threshold(imgC, 0, 255, cv2.THRESH_BINARY)
                    
                    C_points = np.column_stack(np.where(binaryC > 0))
                    if C_points.shape[0] < 3:
                        print(f"⚠  {image_c_path}")
                        continue

    
                    tri = Delaunay(C_points)
                    simplices = tri.simplices
                    edges = []
                    for simplex in simplices:
                      
                        edges.extend([[simplex[i], simplex[(i+1)%3]] for i in range(3)])
                    edges = np.array(edges)
                    edges.sort(axis=1)  
                    edges = np.unique(edges, axis=0) 

                    edge_vectors = C_points[edges[:, 0]] - C_points[edges[:, 1]]
                    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
         
                    if edge_lengths.size == 0:
                        statistics = {
                            "最小值": 0, "最大值": 0, "均值": 0, "中位数": 0, "众数": 0,
                            "25%分位数": 0, "50%分位数": 0, "75%分位数": 0, "95%分位数": 0
                        }
                    else:
                        values, counts = np.unique(edge_lengths, return_counts=True)
                        mode_val = values[np.argmax(counts)] if len(values) > 0 else 0
                        percentiles = np.percentile(edge_lengths, [25, 50, 75, 95])
                        statistics = {
                            "最小值": np.min(edge_lengths),
                            "最大值": np.max(edge_lengths),
                            "均值": np.mean(edge_lengths),
                            "中位数": np.median(edge_lengths),
                            "众数": mode_val,
                            "25%分位数": percentiles[0],
                            "50%分位数": percentiles[1],
                            "75%分位数": percentiles[2],
                            "95%分位数": percentiles[3],
                        }
                    df_statistics = pd.DataFrame([statistics])
                    

                    threshold = R * statistics["均值"] if statistics["均值"] > 0 else 0
                    filtered_mask = edge_lengths <= threshold
                    filtered_edges = edges[filtered_mask]

                    COLORS = {
                        "red": (0, 0, 255),
                        "green": (0, 255, 0),
                        "blue": (255, 0, 0)
                    }
                    img_plot = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
                    for edge in filtered_edges:
                        pt1 = tuple(C_points[edge[0]][::-1].astype(int)) 
                        pt2 = tuple(C_points[edge[1]][::-1].astype(int))
                        cv2.line(img_plot, pt1, pt2,  COLORS[color], 1)

                    output_filtered_path = os.path.join(tree_dir, f'Filtered_E{F}_{R}.png')
                    cv2.imwrite(output_filtered_path, img_plot)
                    
                    

                    img_plot_blue = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
                    for edge in filtered_edges:
                        pt1 = tuple(C_points[edge[0]][::-1].astype(int))
                        pt2 = tuple(C_points[edge[1]][::-1].astype(int))
                        cv2.line(img_plot_blue, pt1, pt2,  COLORS["blue"], 1)
            
                    for pt in C_points:
                        pt_xy = tuple(pt[::-1].astype(int))
                        cv2.circle(img_plot_blue, pt_xy, 1, (255,255,255), -1)
            
                    output_blue_path = os.path.join(show_tree_dir, f'Filtered_E_blue_{F}_{R}.png')
                    cv2.imwrite(output_blue_path, img_plot_blue)
                   
                    
                   
                    df_statistics.to_excel(writer, sheet_name=f'F_{F}', index=False)
            

        
        # ================= 第二部分：异常值处理、图像处理、计算指标 ================= 
        print("\n----- 开始第二部分处理（show与fit文件） -----")
    
        # 
        def remove_outlier_indices(values, m):

            values = np.array(values)
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - m * IQR
            upper_bound = Q3 + m * IQR
            return np.where((values >= lower_bound) & (values <= upper_bound))[0]
    
        
        all_results = []     
        Z_values = []         
        struct_eff_list = []  
        base_act_list = []   
        avg_expr_list = []    
    
        for F in A:
            print(f"Processing F={F} ...")
            
            
            image_c_path = os.path.join(base_path, f'T3-ZLQ {F}% dpi_masks.png')
            imgC = cv2.imread(image_c_path, cv2.IMREAD_GRAYSCALE)
            if imgC is None:
                print(f"⚠ {image_c_path}")
                continue
        
            _, binaryC = cv2.threshold(imgC, 0, 255, cv2.THRESH_BINARY)
            C_points = np.column_stack(np.where(binaryC > 0))
            if C_points.shape[0] == 0:
                print(f"⚠  {image_c_path}")
                continue
    
      
            image_b_path = os.path.join(deal_path, f'{F}%_c2.tif')
            try:
                image_b = Image.open(image_b_path).convert("L")
                image_b_array = np.array(image_b, dtype=np.float32)
            except Exception as e:
                print(f"⚠  {e}")
                continue
           
            B_indices = np.column_stack(np.where(image_b_array > 0))
            B_values = image_b_array[image_b_array > 0]
            if B_values.size == 0:
                print(f" {image_b_path}")
                continue
          
            valid_indices = remove_outlier_indices(B_values, m)
         
            all_indices = np.arange(B_values.size)
            outlier_indices = np.setdiff1d(all_indices, valid_indices)
           
            filtered_B_image = np.zeros_like(image_b_array)
            valid_B_coords = B_indices[valid_indices]
            filtered_B_image[tuple(valid_B_coords.T)] = image_b_array[tuple(valid_B_coords.T)]
            
            
            # image_b_color = cv2.imread(image_b_path, cv2.IMREAD_COLOR)
            # B_image_color = cv2.cvtColor(np.array(image_b_color), cv2.COLOR_GRAY2BGR)
            # outlier_B_coords = B_indices[outlier_indices]
            # for pt in outlier_B_coords:
            #     pt_xy = tuple(pt[::-1].astype(int))
            #     cv2.circle(B_image_color, pt_xy, 1, (0,255,255), -1)
            #
            # output_B_filter = os.path.join(show_dir, f'B_filter_{F}_2.png')
            # cv2.imwrite(output_B_filter, B_image_color)
            #
            
            
            B_image_color = cv2.imread(image_b_path, cv2.IMREAD_COLOR)
            outlier_B_coords = B_indices[outlier_indices]
            for pt in outlier_B_coords:
                pt_xy = tuple(pt[::-1].astype(int))
                cv2.circle(B_image_color, pt_xy, 1, (0,255,255), -1)
           
            output_B_filter = os.path.join(show_dir, f'B_filter_{F}_2.png')
            cv2.imwrite(output_B_filter, B_image_color)
           

    
            #
            highlight_b_array = np.stack([np.zeros_like(filtered_B_image),
                                          filtered_B_image,
                                          np.zeros_like(filtered_B_image)], axis=-1)
   
            binary_colored = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
            if binary_colored.shape != highlight_b_array.shape:
                print(f"Shape mismatch: binaryC.shape={binary_colored.shape}, highlight_b_array.shape={highlight_b_array.shape}")
                highlight_b_array = cv2.resize(highlight_b_array, (binary_colored.shape[1], binary_colored.shape[0]))
                highlight_b_array = highlight_b_array.astype(np.uint8)
            else:
                highlight_b_array = highlight_b_array.astype(np.uint8)
          
            overlay_result = cv2.addWeighted(binary_colored, 1.0, highlight_b_array, 1.0, 0)
            output_d_path = os.path.join(show_dir, f'D_processed_{F}.png')
            cv2.imwrite(output_d_path, overlay_result)

    
            
            D_high = np.zeros_like(filtered_B_image, dtype=np.uint8)

            green_channel = (filtered_B_image > 0).astype(np.uint8)*255
            D_high = cv2.merge([np.zeros_like(green_channel), green_channel, np.zeros_like(green_channel)])
        
            for pt in C_points:
                pt_xy = tuple(pt[::-1].astype(int))
                cv2.circle(D_high, pt_xy, 1, (255,255,255), -1)
            output_D_high = os.path.join(show_dir, f'D_high_{F}.png')
            cv2.imwrite(output_D_high, D_high)
          
  
            output_e_path = os.path.join(tree_dir, f'Filtered_E{F}_{R}.png')
            imgD = cv2.imread(output_d_path, cv2.IMREAD_COLOR)
            imgE = cv2.imread(output_e_path, cv2.IMREAD_COLOR)
            if imgD is None or imgE is None:
                print(f"⚠ : {output_d_path} 或 {output_e_path}")
                continue
            
            grayD = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)
            grayE = cv2.cvtColor(imgE, cv2.COLOR_BGR2GRAY)
            D_points = np.column_stack(np.where(grayD > 0))
            E_points = np.column_stack(np.where(grayE > 0))
            if D_points.shape[0] == 0 or E_points.shape[0] == 0:
                print(f"⚠: F={F}")
                continue
    

            C_set = set(map(tuple, C_points))
            D_prime = np.array([pt for pt in D_points if tuple(pt) not in C_set])
            E_prime = np.array([pt for pt in E_points if tuple(pt) not in C_set])
            D_prime_set = set(map(tuple, D_prime))
            E_prime_set = set(map(tuple, E_prime))
            X_set = D_prime_set & E_prime_set
            X = np.array(list(X_set))
            
            image_a_path = os.path.join(deal_path, f'{F}%_c1.tif')
            imgA = cv2.imread(image_a_path, cv2.IMREAD_COLOR)
            if imgA is None:
                print(f"⚠: {image_a_path}")
                continue
            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            A_indices = np.column_stack(np.where(grayA > 0))
            A_values = grayA[grayA > 0]
            if A_values.size == 0:
                print(f"⚠: F={F}")
                continue
            valid_indices_A = remove_outlier_indices(A_values, m)

            all_indices_A = np.arange(A_values.size)
            outlier_indices_A = np.setdiff1d(all_indices_A, valid_indices_A)
            A_filtered_points = A_indices[valid_indices_A]
            A_filtered_set = set(map(tuple, A_filtered_points))
            Y_set = A_filtered_set & C_set
            Y = np.array(list(Y_set))
            
            #
            A_image_color = imgA.copy()
            outlier_A_coords = A_indices[outlier_indices_A]
            for pt in outlier_A_coords:
                pt_xy = tuple(pt[::-1].astype(int))
                cv2.circle(A_image_color, pt_xy, 1, (0,255,255), -1)
            output_A_filter = os.path.join(show_dir, f'A_filter_{F}.png')
            cv2.imwrite(output_A_filter, A_image_color)
           
    

            if len(X_set) > 0:
                avg_expression = np.mean([filtered_B_image[pt[0], pt[1]] for pt in X_set])
            else:
                avg_expression = 0
            print(f"avg_expression = {avg_expression}")
            
            if len(E_prime) > 0 and len(C_set) > 0:
                structural_efficiency = len(X) / len(E_prime)
                baseline_activity_ratio = len(Y) / len(C_set)
            else:
                structural_efficiency = 0
                baseline_activity_ratio = 0
                
            Z = structural_efficiency * baseline_activity_ratio * avg_expression
            Z_values.append(Z)
            struct_eff_list.append(structural_efficiency)
            base_act_list.append(baseline_activity_ratio)
            avg_expr_list.append(avg_expression)
            
     
            # A_filtered_image = np.zeros_like(grayA)
            # A_filtered_image[tuple(A_filtered_points.T)] = grayA[tuple(A_filtered_points.T)]
            # A_filtered_color = cv2.cvtColor(A_filtered_image, cv2.COLOR_GRAY2BGR)
            # A_high = np.where(A_filtered_image[..., None] > 0, (255,0,255), A_filtered_color)
            # A_high = A_high.astype(np.uint8)
            # C_overlay = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
            # overlay_A_high = cv2.addWeighted(A_high, 0.5, C_overlay, 0.5, 0)
            # output_A_high = os.path.join(show_dir, f'A_high_{F}.png')
            # cv2.imwrite(output_A_high, overlay_A_high)

            
            
            # A_C=A_filtered_color.astype(np.uint8)
            # overlay_A_C = cv2.addWeighted(A_C, 0.5, C_overlay, 0.5, 0)
            # output_A_C = os.path.join(show_dir, f'A_C_{F}.png')
            # cv2.imwrite(output_A_C, overlay_A_C)

            
            
            
            
    
            A_filtered_image = np.zeros_like(grayA)
            A_filtered_image[tuple(A_filtered_points.T)] = grayA[tuple(A_filtered_points.T)]


            A_filtered_color = cv2.cvtColor(A_filtered_image, cv2.COLOR_GRAY2BGR)

            A_high = np.where(A_filtered_image[..., None] > 0, (255, 0, 255), A_filtered_color)
            A_high = A_high.astype(np.uint8)

            C_overlay = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
            overlay_A_high = cv2.addWeighted(A_high, 0.5, C_overlay, 0.5, 0)
            output_A_high = os.path.join(show_dir, f'A_high_{F}.png')
            cv2.imwrite(output_A_high, overlay_A_high)
            
   
            A_C = np.zeros_like(imgA) 
            A_C[tuple(A_filtered_points.T)] = imgA[tuple(A_filtered_points.T)]


            overlay_A_C = cv2.addWeighted(A_C, 0.5, C_overlay, 0.5, 0)
            output_A_C = os.path.join(show_dir, f'A_C_{F}_2.png')
            cv2.imwrite(output_A_C, overlay_A_C)
            

        
            all_results.append({
                "F": F,
                "R": R,
                "SIF": round(Z, 4),
                "structural_efficiency": round(structural_efficiency, 4),
                "baseline_activity_ratio": round(baseline_activity_ratio, 4),
                "avg_expression": avg_expression,
                "r_value": None,
                "p_value": None
            })

        #################################################
        try:
            A_np = np.array(A, dtype=float)
            is_numeric = True
        except:
            is_numeric = False
    
        if is_numeric and len(Z_values) == len(A):

            Z_values_np = np.array(Z_values)

            slope, intercept, r_value, p_value, _ = stats.linregress(Z_values_np, A_np)

            all_results.append({
                "F": "N/A",
                "R": R,
                "Fit": f"Y = {slope:.4f} * Z + {intercept:.4f}",
                "r_value": round(r_value, 4),
                "p_value": round(p_value, 4)
            })

            plt.figure(figsize=(6, 4))
            plt.scatter(Z_values_np, A_np, color='blue', label='Data')
            plt.plot(Z_values_np, slope * Z_values_np + intercept, color='red', linewidth=2,
                     label=f'Fit: Y={slope:.2f}Z+{intercept:.2f}\nr={r_value:.2e}')
            plt.xlabel('Structural Inhibitory Force (SIF)')
            plt.ylabel('Residual Viable Tumor (%)')
            plt.title(f'R={R}: SIF positively correlated with RVT')
            plt.legend()
            plt.grid()
            fit_file = os.path.join(fit_dir, f'fit_{R}.pdf')
            plt.savefig(fit_file)
            plt.close()

        else:
            plt.figure(figsize=(6, 4))
            plt.plot(A, Z_values, marker='o', linestyle='-', color='blue', label='SIF')
            plt.xlabel('Sample')
            plt.ylabel('Structural Inhibitory Force (SIF)')
            plt.title(f'R={R}: SIF Quantization')
            plt.legend()
            plt.grid()
            line_file = os.path.join(fit_dir, f'line_{R}.pdf')
            plt.savefig(line_file)
            plt.close()
  
  
        plt.figure(figsize=(8, 5))
        if is_numeric:
            x_values = A_np
        else:
            x_values = range(len(A))
            plt.xticks(x_values, A)
        plt.plot(x_values, struct_eff_list, marker='o', label='Structural Efficiency')
        plt.plot(x_values, base_act_list, marker='^', label='Baseline Activity Ratio')
        plt.xlabel('Sample')
        plt.ylabel('Values')
        plt.title(f'R={R}: Structural Efficiency & Baseline Activity Ratio')
        plt.legend()
        plt.grid()
        lineplot1_file = os.path.join(fit_dir, f'lineplot_se_bar_{R}.pdf')
        plt.savefig(lineplot1_file)
        plt.close()

        plt.figure(figsize=(8, 5))
        if is_numeric:
            x_values = A_np
        else:
            x_values = range(len(A))
            plt.xticks(x_values, A)
        plt.plot(x_values, avg_expr_list, marker='s', label='Average Expression', color='green')
        plt.xlabel('Sample')
        plt.ylabel('Average Expression Value')
        plt.title(f'R={R}: Average Expression')
        plt.legend()
        plt.grid()
        lineplot2_file = os.path.join(fit_dir, f'lineplot_avg_expr_{R}.pdf')
        plt.savefig(lineplot2_file)
        plt.close()

        df_results = pd.DataFrame(all_results)

        df_results.to_excel(writer_all, sheet_name=f'R_{R}', index=False)
        print(f"\n 当前R={R}的结果已写入综合Excel的Sheet: R_{R}")
    

print(f"\n 所有R的综合结果已保存到 {excel_show_all_path}")    
