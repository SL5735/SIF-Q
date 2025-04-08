import cv2 
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.spatial import Delaunay
from scipy import stats
import numpy as np
from PIL import Image

# ================= 参数设置 =================
color = "green"  
# 设定 m 与 R 的取值列表
m_values = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3]
R_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150]

# 版本号
version = f"{color}-mRr-de3.0"

# 其他参数
file = 'mice-3E-Native'
tree_name = 'treecut-unique'
count = f"{file}_{color}_{tree_name}_T_E"

A = np.array([100, 50, 10, 5, 0])

# ================= 路径设置 =================

base_path = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\mice\3E\Native\ICF-dpienhanced'
deal_path = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\mice\3E\Native'


output_dir = os.path.join(deal_path, f"try_{version}")
os.makedirs(output_dir, exist_ok=True)


exp_R_fixed_dir = os.path.join(output_dir, "R_fixed")
exp_m_fixed_dir = os.path.join(output_dir, "m_fixed")
os.makedirs(exp_R_fixed_dir, exist_ok=True)
os.makedirs(exp_m_fixed_dir, exist_ok=True)

# ================= 辅助函数 =================
def remove_outlier_indices(values, m_coef):

    values = np.array(values)
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - m_coef * IQR
    upper_bound = Q3 + m_coef * IQR
    return np.where((values >= lower_bound) & (values <= upper_bound))[0]

def run_analysis(m, R, save_folder):


    tree_dir = os.path.join(deal_path, f"{color}_{tree_name}")
    show_dir = os.path.join(save_folder, count)
    fit_dir = os.path.join(save_folder, f"fit{count}")
    for d in [tree_dir, show_dir, fit_dir]:
        os.makedirs(d, exist_ok=True)
        
    # ---------------- tree ----------------

    excel_tree_path = os.path.join(tree_dir, f'edge_analysis_{color}_{R}.xlsx')
    if os.path.exists(excel_tree_path):
        print(f"已存在 tree 文件，跳过第一部分处理")
    else:
        print(f"----- 第一部分处理（tree文件）: R={R}, m={m} -----")
        with pd.ExcelWriter(excel_tree_path, engine='xlsxwriter') as writer:
            for F in A:
                print(f"Processing F={F} ...")
                # 读取 C 图（灰度图）
                image_c_path = os.path.join(base_path, f'P3-ZLQ {F}% dpi_masks.png')
                imgC = cv2.imread(image_c_path, cv2.IMREAD_GRAYSCALE)
                if imgC is None:
                    print(f"C 图加载失败: {image_c_path}")
                    continue

                # 二值化处理
                _, binaryC = cv2.threshold(imgC, 0, 255, cv2.THRESH_BINARY)
                # 获取所有非零点（注意 np.where 返回 (row, col)）
                C_points = np.column_stack(np.where(binaryC > 0))
                if C_points.shape[0] < 3:
                    print(f"{image_c_path}")
                    continue

                # Delaunay 三角剖分
                tri = Delaunay(C_points)
                simplices = tri.simplices
                edges = []
                for simplex in simplices:
                    edges.extend([[simplex[i], simplex[(i+1)%3]] for i in range(3)])
                edges = np.array(edges)
                edges.sort(axis=1)
                edges = np.unique(edges, axis=0)

                # 计算边长
                edge_vectors = C_points[edges[:, 0]] - C_points[edges[:, 1]]
                edge_lengths = np.linalg.norm(edge_vectors, axis=1)
                
                if edge_lengths.size == 0:
                    statistics = {"最小值": 0, "最大值": 0, "均值": 0, "中位数": 0, "众数": 0,
                                  "25%分位数": 0, "50%分位数": 0, "75%分位数": 0, "95%分位数": 0}
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
                

                COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
                img_plot = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
                for edge in filtered_edges:
                    pt1 = tuple(C_points[edge[0]][::-1].astype(int))
                    pt2 = tuple(C_points[edge[1]][::-1].astype(int))
                    cv2.line(img_plot, pt1, pt2, COLORS[color], 1)
                

                output_filtered_path = os.path.join(tree_dir, f'Filtered_E{F}_{R}_{color}.png')
                cv2.imwrite(output_filtered_path, img_plot)
                print(f"{output_filtered_path}")
                
 
                df_statistics.to_excel(writer, sheet_name=f'F_{F}', index=False)

        print("第一部分处理完成！")
    
    # ---------------- 第二部分----------------
    print(f"----- 第二部分处理（show与fit文件）: R={R}, m={m} -----")
    all_results = []  
    Z_values = []     
    se_list = []      
    tce_list = []  
    bar_list = []     
    
    # T 为常量参数，如无特殊要求可设为1
    T = 1  

    for F in A:
        print(f"Processing F={F} ...")
 
        image_c_path = os.path.join(base_path, f'P3-ZLQ {F}% dpi_masks.png')
        imgC = cv2.imread(image_c_path, cv2.IMREAD_GRAYSCALE)
        if imgC is None:
            print(f"C 图加载失败: {image_c_path}")
            continue
        _, binaryC = cv2.threshold(imgC, 0, 255, cv2.THRESH_BINARY)
        C_points = np.column_stack(np.where(binaryC > 0))
        if C_points.shape[0] == 0:
            print(f"{image_c_path}")
            continue
        C_set = set(map(tuple, C_points))
        

        image_b1_path = os.path.join(deal_path, f'{F}% Native_c1.tif')
        try:
            image_b1 = Image.open(image_b1_path).convert("L")
            image_b1_array = np.array(image_b1, dtype=np.float32)
        except Exception as e:
            print(f"B1 图加载失败: {e}")
            continue

        image_b2_path = os.path.join(deal_path, f'{F}% Native_c3.tif')
        try:
            image_b2 = Image.open(image_b2_path).convert("L")
            image_b2_array = np.array(image_b2, dtype=np.float32)
        except Exception as e:
            print(f"B2 图加载失败: {e}")
            continue

        B1_points_all = np.column_stack(np.where(image_b1_array > 0))
        B2_points_all = np.column_stack(np.where(image_b2_array > 0))
        if B1_points_all.shape[0] == 0 or B2_points_all.shape[0] == 0:
            print(f" F={F}")
            continue

        B1_values = [image_b1_array[pt[0], pt[1]] for pt in B1_points_all]
        B2_values = [image_b2_array[pt[0], pt[1]] for pt in B2_points_all]
        indices_B1 = remove_outlier_indices(B1_values, m)
        indices_B2 = remove_outlier_indices(B2_values, m)
        B1_points_filtered = B1_points_all[indices_B1]
        B2_points_filtered = B2_points_all[indices_B2]
        B1_set_filtered = set(map(tuple, B1_points_filtered))
        B2_set_filtered = set(map(tuple, B2_points_filtered))
        B_intersect_set = B1_set_filtered & B2_set_filtered


        output_e_path = os.path.join(tree_dir, f'Filtered_E{F}_{R}_{color}.png')
        imgE = cv2.imread(output_e_path, cv2.IMREAD_COLOR)
        if imgE is None:
            print(f" E 图加载失败: {output_e_path}")
            continue
        grayE = cv2.cvtColor(imgE, cv2.COLOR_BGR2GRAY)
        E_points_all = np.column_stack(np.where(grayE > 0))
        if E_points_all.shape[0] == 0:
            print(f"F={F}")
            continue
        E_set = set(map(tuple, E_points_all))
        Eprime_set = E_set - C_set


        Dprime_set = B_intersect_set - C_set
        DE_intersection = Dprime_set & Eprime_set
        if len(Eprime_set) > 0:
            structural_efficiency = len(DE_intersection) / len(Eprime_set)
        else:
            structural_efficiency = 0
            print(f"F={F}")
        print(f"F={F} structural_efficiency: {structural_efficiency}")


        image_a_path = os.path.join(deal_path, f'{F}% Native_c2.tif')
        imgA = cv2.imread(image_a_path, cv2.IMREAD_COLOR)
        if imgA is None:
            print(f" A 图加载失败: {image_a_path}")
            continue
        grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        A_points = np.column_stack(np.where(grayA > 0))
        if A_points.shape[0] == 0:
            print(f"F={F}")
            continue
        A_values = [image_b1_array[pt[0], pt[1]] for pt in A_points]
        indices_A = remove_outlier_indices(A_values, m)
        A_points_filtered = A_points[indices_A]
        A_prime_set = set(map(tuple, A_points_filtered))
        Y_set = A_prime_set & C_set
        baseline_activity_ratio = len(Y_set) / len(C_set)
        print(f"F={F} baseline_activity_ratio: {baseline_activity_ratio}")


        overlap_points = np.array(list(DE_intersection))
        if overlap_points.shape[0] > 0:
            B1_overlap = image_b1_array[overlap_points[:, 0], overlap_points[:, 1]]
            B2_overlap = image_b2_array[overlap_points[:, 0], overlap_points[:, 1]]
            avg_expression = np.mean((B1_overlap + B2_overlap) / 2)
            if np.std(B1_overlap) > 0 and np.std(B2_overlap) > 0:
                corr_coef = np.corrcoef(B1_overlap, B2_overlap)[0, 1]
            else:
                corr_coef = 0
            total_co_expression = avg_expression * corr_coef
        else:
            total_co_expression = 0

        Z = structural_efficiency * total_co_expression * baseline_activity_ratio
        print(f"F={F} total_co_expression: {total_co_expression:.2f}")

        # 记录结果
        all_results.append({
            "sample": F,
            "R": R,
            "m": m,
            "SIF": round(Z, 4),
            "structural_efficiency": round(structural_efficiency, 4),
            "total_co_expression": round(total_co_expression, 4),
            "baseline_activity_ratio": round(baseline_activity_ratio, 4)
        })
        Z_values.append(Z)
        se_list.append(structural_efficiency)
        tce_list.append(total_co_expression)
        bar_list.append(baseline_activity_ratio)
    
    # ---------------- 回归分析与拟合图 ----------------
    try:
        A_np = np.array(A, dtype=float)
        is_numeric = True
    except:
        is_numeric = False

    r_value_out = None
    if is_numeric and len(Z_values) == len(A):
        Z_values_np = np.array(Z_values)
        slope, intercept, r_value, p_value, _ = stats.linregress(Z_values_np, A_np)
        r_value_out = r_value
        all_results.append({
            "R": R,
            "m": m,
            "Fit": f"RVT = {slope:.4f} * SIF + {intercept:.4f}",
            "r_value": round(r_value, 4),
            "p_value": round(p_value, 4)
        })
        plt.figure(figsize=(6, 4))
        plt.scatter(Z_values_np, A_np, color='blue', label='Data')
        plt.plot(Z_values_np, slope * Z_values_np + intercept, color='red', linewidth=2,
                 label=f'Fit: {slope:.2f}*SIF+{intercept:.2f}\nr={r_value:.2e}')
        plt.xlabel('Structural Inhibitory Force (SIF)')
        plt.ylabel('Residual Viable Tumor (%)')
        plt.title(f'R={R}, m={m}: SIF vs RVT')
        plt.legend()
        plt.grid()
        fit_file = os.path.join(fit_dir, f'fit_R{R}_m{m}.pdf')
        plt.savefig(fit_file)
        plt.close()
        print(f"拟合图保存: {fit_file}")
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(A, Z_values, marker='o', linestyle='-', color='blue', label='SIF')
        plt.xlabel('Sample')
        plt.ylabel('SIF')
        plt.title(f'R={R}, m={m}: SIF Quantization')
        plt.legend()
        plt.grid()
        line_file = os.path.join(fit_dir, f'line_R{R}_m{m}.pdf')
        plt.savefig(line_file)
        plt.close()
        print(f"折线图保存: {line_file}")
    
    # 此外，绘制结构效率、baseline_activity_ratio、total_co_expression 的折线图（可选）
    plt.figure(figsize=(8, 5))
    x_values = A_np if is_numeric else range(len(A))
    plt.plot(x_values, se_list, marker='o', label='Structural Efficiency')
    plt.plot(x_values, bar_list, marker='^', label='Baseline Activity Ratio')
    plt.xlabel('Sample')
    plt.ylabel('Values')
    plt.title(f'R={R}, m={m}: SE & BAR')
    plt.legend()
    plt.grid()
    se_bar_file = os.path.join(fit_dir, f'se_bar_R{R}_m{m}.pdf')
    plt.savefig(se_bar_file)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, tce_list, marker='s', label='Total Co-expression', color='green')
    plt.xlabel('Sample')
    plt.ylabel('Total Co-expression')
    plt.title(f'R={R}, m={m}: Total Co-expression')
    plt.legend()
    plt.grid()
    tce_file = os.path.join(fit_dir, f'tce_R{R}_m{m}.pdf')
    plt.savefig(tce_file)
    plt.close()
    

    excel_show_path = os.path.join(show_dir, f'analysis_results_R{R}_m{m}.xlsx')
    with pd.ExcelWriter(excel_show_path, engine='openpyxl') as writer:
        df_results = pd.DataFrame(all_results)
        sheet_name = f"R{R}_m{m}"
        df_results.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"综合结果保存到 Excel: Sheet {sheet_name}")
    
    return r_value_out

# ================= 运行实验 =================

R_fixed_results = {}  # {R: [(m, r_value), ...]}
for R in R_values:
    folder_R = os.path.join(exp_R_fixed_dir, f"R_{R}")
    os.makedirs(folder_R, exist_ok=True)
    m_list = []
    for m in m_values:
        folder_m = os.path.join(folder_R, f"m_{m}")
        os.makedirs(folder_m, exist_ok=True)
        print(f"\n===== R 固定实验：R={R}, m={m} =====")
        r_val = run_analysis(m, R, folder_m)
        if r_val is not None:
            m_list.append((m, r_val))
    R_fixed_results[R] = m_list
    #
    if m_list:
        m_vals, r_vals = zip(*m_list)
        plt.figure(figsize=(8, 5))
        plt.plot(m_vals, r_vals, marker='o', linestyle='-')
        plt.xlabel('m')
        plt.ylabel('r_value')
        plt.title(f'R={R}: m vs r_value')
        plt.ylim(0, 1)
        plt.grid()
        plot_path = os.path.join(folder_R, f"m_vs_r_value_R{R}.pdf")
        plt.savefig(plot_path)
        plt.close()
        print(f"m vs r_value 折线图保存: {plot_path}")


m_fixed_results = {}  # {m: [(R, r_value), ...]}
for m in m_values:
    folder_m = os.path.join(exp_m_fixed_dir, f"m_{m}")
    os.makedirs(folder_m, exist_ok=True)
    R_list = []
    for R in R_values:
        folder_R = os.path.join(folder_m, f"R_{R}")
        os.makedirs(folder_R, exist_ok=True)
        print(f"\n===== m 固定实验：m={m}, R={R} =====")
        r_val = run_analysis(m, R, folder_R)
        if r_val is not None:
            R_list.append((R, r_val))
    m_fixed_results[m] = R_list

    if R_list:
        R_vals, r_vals = zip(*R_list)
        plt.figure(figsize=(8, 5))
        plt.plot(R_vals, r_vals, marker='o', linestyle='-')
        plt.xlabel('R')
        plt.ylabel('r_value')
        plt.title(f'm={m}: R vs r_value')
        plt.ylim(0, 1)
        plt.grid()
        plot_path = os.path.join(folder_m, f"R_vs_r_value_m{m}.pdf")
        plt.savefig(plot_path)
        plt.close()
        print(f"R vs r_value 折线图保存: {plot_path}")

print("\n 所有实验运行完毕！")
