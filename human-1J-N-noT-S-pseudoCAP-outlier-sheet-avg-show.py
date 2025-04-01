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
m = 2  ###########去除outlier的多少
version = f"{color}-noT-S-pseudoCAP-m{m}-de3.0-avg_show"  # 可修改版本号

# 【修改】增加R循环变量列表，原来固定R=10改为循环（可根据需要修改R_values）
R_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150]  # 示例R值

A = np.array([95, 60, 20, 4, 0])
T_values = np.arange(1.0, 1.3, 0.1)  # T 从 1.0 到 1.2，步长 0.1

# 其他参数
file = 'human-1J-N'
tree = f'treecut-unique'  # 【修改】不在文件夹名中添加R
count = f"{file}_{color}_{tree}_T_E"  # 【修改】同上

# ================= 路径设置 =================
# 原始图片所在路径（保证图片路径正确）
base_path = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\human\1J-N\human_dpienhanced'
deal_path = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\human\1J-N'

# 整合输出目录：所有输出文件将存放于 try_{version} 文件夹下
output_dir = os.path.join(deal_path, f"try_{version}")
os.makedirs(output_dir, exist_ok=True)

# 在 output_dir 下建立三个子文件夹：tree（保存第一部分结果）、show（保存第二部分部分结果与excel）、fit（保存拟合图）
tree_dir = os.path.join(deal_path, f"{color}_{tree}")
show_dir = os.path.join(output_dir, count)
fit_dir = os.path.join(output_dir, f"fit{count}")

for path in [tree_dir, show_dir, fit_dir]:
    os.makedirs(path, exist_ok=True)

# 新增：第一部分额外保存路径，专用于保存蓝色边与白色C图前景点覆盖后的结果
show_tree_dir = r'C:\Users\User\Desktop\test_cns\ALL-tree-T\human\1J-N\show-tree'
os.makedirs(show_tree_dir, exist_ok=True)

###############################################
# ================= 新增：对每个 F（A中的每个元素）读取 B 图并保存统计信息 =================
excel_B_stats_path = os.path.join(deal_path, f'B_stats_{color}_{tree}.xlsx')
if not os.path.exists(excel_B_stats_path):   # 【NEW】如果已存在，则跳过B图统计部分
    with pd.ExcelWriter(excel_B_stats_path, engine='xlsxwriter') as writer:
        for F in A:
            # B 图文件名，按照原代码中CD的规则，C1对应B图
            image_b_path = os.path.join(deal_path, f'{F}%_c2.tif')
            try:
                image_b = Image.open(image_b_path).convert("L")
                image_b_array = np.array(image_b, dtype=np.float32)
            except Exception as e:
                print(f"⚠ B 图加载失败: {image_b_path}，错误信息: {e}")
                continue

            # 这里选取非零像素（也可以根据需要对全部矩阵统计）
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
                # 计算众数（使用scipy.stats.mode，注意新版scipy返回ModeResult对象）
                mode_result = stats.mode(positive_values, nan_policy='omit')
                mode_val = np.atleast_1d(mode_result.mode)[0] if np.atleast_1d(mode_result.count)[0] > 0 else None
                percentiles = np.percentile(positive_values, [25, 50, 75, 95])
                # 95-100位点的值区间：这里取 95%分位数 到最大值
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
            print(f"✅ B 图统计数据已保存到 Sheet: {sheet_name}")
    print(f"✅ 所有 B 图统计数据已保存到 {excel_B_stats_path}")
else:
    print(f"✅ 检测到已存在的 B 图统计文件，跳过统计处理: {excel_B_stats_path}")

# ================= 定义综合保存第二部分结果的Excel文件 =================
# 此处修改：不再每个R单独保存一个Excel文件，而是将每个R结果写入同一Excel文件的不同sheet中
excel_show_all_path = os.path.join(show_dir, f'analysis_results_{count}.xlsx')

# 创建ExcelWriter对象，用于保存所有R的结果到不同的sheet
with pd.ExcelWriter(excel_show_all_path, engine='openpyxl') as writer_all:

    # ##########===== 外层循环：对每个R值进行处理 =====
    for R in R_values:
        print(f"\n==== 开始处理 R = {R} ====")
        
        # 【修改】Excel保存路径中添加R标识，仅用于第一部分tree文件（独立保存）
        excel_tree_path = os.path.join(tree_dir, f'edge_analysis_{color}_{R}.xlsx')
        # ================= 第一部分：生成 tree 文件（Delaunay 三角剖分及边过滤） =================
        # 【NEW】如果该R对应的tree结果已存在，则跳过第一部分的计算
        if os.path.exists(excel_tree_path):
            print(f"✅ 检测到已存在的第一部分结果文件，跳过R={R}的第一部分处理: {excel_tree_path}")
        else:
            print("----- 开始第一部分处理（tree文件） -----")
            with pd.ExcelWriter(excel_tree_path, engine='xlsxwriter') as writer:
                for F in A:
                    print(f"Processing F={F} ...")
                    
                    # 读取 C 图（灰度图）
                    image_c_path = os.path.join(base_path, f'T3-ZLQ {F}% dpi_masks.png')
                    imgC = cv2.imread(image_c_path, cv2.IMREAD_GRAYSCALE)
                    if imgC is None:
                        print(f"⚠ C 图加载失败: {image_c_path}")
                        continue

                    # 二值化处理
                    _, binaryC = cv2.threshold(imgC, 0, 255, cv2.THRESH_BINARY)
                    # 获取所有非零点（注意：np.where 返回 (row, col) 顺序）
                    C_points = np.column_stack(np.where(binaryC > 0))
                    if C_points.shape[0] < 3:
                        print(f"⚠ C 图点数不足，无法计算 Delaunay: {image_c_path}")
                        continue

                    # Delaunay 三角剖分
                    tri = Delaunay(C_points)
                    simplices = tri.simplices
                    edges = []
                    for simplex in simplices:
                        # 每个三角形有 3 条边
                        edges.extend([[simplex[i], simplex[(i+1)%3]] for i in range(3)])
                    edges = np.array(edges)
                    edges.sort(axis=1)  # 按顶点索引排序
                    edges = np.unique(edges, axis=0)  # 去重

                    # 计算所有边的长度
                    edge_vectors = C_points[edges[:, 0]] - C_points[edges[:, 1]]
                    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
                    
                    # 统计计算
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
                    
                    # 设定剪枝阈值，并过滤边
                    threshold = R * statistics["均值"] if statistics["均值"] > 0 else 0
                    filtered_mask = edge_lengths <= threshold
                    filtered_edges = edges[filtered_mask]
                    # 定义颜色字典（BGR格式）
                    COLORS = {
                        "red": (0, 0, 255),
                        "green": (0, 255, 0),
                        "blue": (255, 0, 0)
                    }

                    # 在原图上绘制过滤后的边（原版，用当前color绘制）
                    img_plot = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
                    for edge in filtered_edges:
                        pt1 = tuple(C_points[edge[0]][::-1].astype(int))  # (row,col)转换为(x,y)
                        pt2 = tuple(C_points[edge[1]][::-1].astype(int))
                        cv2.line(img_plot, pt1, pt2,  COLORS[color], 1)
                    # 保存原版过滤结果
                    output_filtered_path = os.path.join(tree_dir, f'Filtered_E{F}_{R}.png')
                    cv2.imwrite(output_filtered_path, img_plot)
                    print(f"✅ Filtered_E 文件已保存: {output_filtered_path}")
                    
                    # 【新增】复制一份，使用蓝色边，并在剪枝后将 C 图的前景点以白色覆盖
                    img_plot_blue = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
                    for edge in filtered_edges:
                        pt1 = tuple(C_points[edge[0]][::-1].astype(int))
                        pt2 = tuple(C_points[edge[1]][::-1].astype(int))
                        cv2.line(img_plot_blue, pt1, pt2,  COLORS["blue"], 1)
                    # 将 C 图前景点以白色覆盖
                    for pt in C_points:
                        pt_xy = tuple(pt[::-1].astype(int))
                        cv2.circle(img_plot_blue, pt_xy, 1, (255,255,255), -1)
                    # 保存新生成的图到 show_tree_dir
                    output_blue_path = os.path.join(show_tree_dir, f'Filtered_E_blue_{F}_{R}.png')
                    cv2.imwrite(output_blue_path, img_plot_blue)
                    print(f"✅ Filtered_E_blue 文件已保存: {output_blue_path}")
                    
                    # 将统计数据写入 Excel 的对应 sheet 中
                    df_statistics.to_excel(writer, sheet_name=f'F_{F}', index=False)
                    print(f"✅ 统计数据已写入 Excel (Sheet: F_{F})")

            print("🎉 第一部分处理完成！")
        
        # ================= 第二部分：异常值处理、图像处理、计算指标 ================= 
        print("\n----- 开始第二部分处理（show与fit文件） -----")
    
        # 定义去除异常值的函数：利用四分位数方法（IQR）计算上下界限，返回满足条件的索引列表
        def remove_outlier_indices(values, m):
            """
            利用四分位数方法计算上下界限，返回满足 [Q1 - m*IQR, Q3 + m*IQR] 条件的索引列表
            :param values: 待检测的一维数组
            :param m: 控制异常值剔除的系数（默认1.5）
            :return: 符合条件的索引数组
            """
            values = np.array(values)
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - m * IQR
            upper_bound = Q3 + m * IQR
            return np.where((values >= lower_bound) & (values <= upper_bound))[0]
    
        # 用于记录每个 F 的计算结果（当前R的结果）
        all_results = []      # 保存 F、R、SIF 以及其他指标
        Z_values = []         # 用于存储每个 F 对应的 SIF（结构指标）值，用于拟合
        struct_eff_list = []  # 存储每个 F 的 structural_efficiency
        base_act_list = []    # 存储每个 F 的 baseline_activity_ratio
        avg_expr_list = []    # 存储每个 F 的 X_set 平均表达量
    
        for F in A:
            print(f"Processing F={F} ...")
            
            # 1. 读取 C 图（灰度图），二值化后提取前景点（C点集）
            image_c_path = os.path.join(base_path, f'T3-ZLQ {F}% dpi_masks.png')
            imgC = cv2.imread(image_c_path, cv2.IMREAD_GRAYSCALE)
            if imgC is None:
                print(f"⚠ C 图加载失败: {image_c_path}")
                continue
            # 二值化（阈值0，所有大于0的像素视为前景）
            _, binaryC = cv2.threshold(imgC, 0, 255, cv2.THRESH_BINARY)
            C_points = np.column_stack(np.where(binaryC > 0))
            if C_points.shape[0] == 0:
                print(f"⚠ C 图未检测到前景点: {image_c_path}")
                continue
    
            # 2. 读取 B 图（灰度图），并利用 IQR 方法去除异常值
            image_b_path = os.path.join(deal_path, f'{F}%_c2.tif')
            try:
                image_b = Image.open(image_b_path).convert("L")
                image_b_array = np.array(image_b, dtype=np.float32)
            except Exception as e:
                print(f"⚠ B 图加载失败: {e}")
                continue
            # 提取 B 图中所有非零（前景）像素及其坐标
            B_indices = np.column_stack(np.where(image_b_array > 0))
            B_values = image_b_array[image_b_array > 0]
            if B_values.size == 0:
                print(f"⚠ B 图未检测到前景点: {image_b_path}")
                continue
            # 得到有效像素的索引
            valid_indices = remove_outlier_indices(B_values, m)
            # 同时获取被剔除的异常点索引（取反）
            all_indices = np.arange(B_values.size)
            outlier_indices = np.setdiff1d(all_indices, valid_indices)
            # 构造过滤后的 B 图：仅保留有效像素，其余置为0
            filtered_B_image = np.zeros_like(image_b_array)
            valid_B_coords = B_indices[valid_indices]
            filtered_B_image[tuple(valid_B_coords.T)] = image_b_array[tuple(valid_B_coords.T)]
            
            # # 【新增】在原 B 图上，将被剔除的异常点标记为黄色（BGR: (0,255,255)）
            # image_b_color = cv2.imread(image_b_path, cv2.IMREAD_COLOR)
            # B_image_color = cv2.cvtColor(np.array(image_b_color), cv2.COLOR_GRAY2BGR)
            # outlier_B_coords = B_indices[outlier_indices]
            # for pt in outlier_B_coords:
            #     pt_xy = tuple(pt[::-1].astype(int))
            #     cv2.circle(B_image_color, pt_xy, 1, (0,255,255), -1)
            # # 保存 B 图过滤异常后的图像（fliter图），保存为高清png
            # output_B_filter = os.path.join(show_dir, f'B_filter_{F}_2.png')
            # cv2.imwrite(output_B_filter, B_image_color)
            # print(f"✅ B_filter 文件已保存: {output_B_filter}")
            
            # 【修改】直接读取彩色的 B 图，并在上面标记异常点为黄色（BGR: (0,255,255)）
            B_image_color = cv2.imread(image_b_path, cv2.IMREAD_COLOR)
            outlier_B_coords = B_indices[outlier_indices]
            for pt in outlier_B_coords:
                pt_xy = tuple(pt[::-1].astype(int))
                cv2.circle(B_image_color, pt_xy, 1, (0,255,255), -1)
            # 保存 B 图过滤异常后的图像（filter图），保存为高清png
            output_B_filter = os.path.join(show_dir, f'B_filter_{F}_2.png')
            cv2.imwrite(output_B_filter, B_image_color)
            print(f"✅ B_filter 文件已保存: {output_B_filter}")

    
            # 3. 利用过滤后的 B 图构造高亮区域（仅在 G 通道显示），并与 C 图叠加生成 D 图
            highlight_b_array = np.stack([np.zeros_like(filtered_B_image),
                                          filtered_B_image,
                                          np.zeros_like(filtered_B_image)], axis=-1)
            # 将 C 图二值化结果转换为彩色图
            binary_colored = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
            if binary_colored.shape != highlight_b_array.shape:
                print(f"Shape mismatch: binaryC.shape={binary_colored.shape}, highlight_b_array.shape={highlight_b_array.shape}")
                highlight_b_array = cv2.resize(highlight_b_array, (binary_colored.shape[1], binary_colored.shape[0]))
                highlight_b_array = highlight_b_array.astype(np.uint8)
            else:
                highlight_b_array = highlight_b_array.astype(np.uint8)
            # 叠加图像：得到 D 图
            overlay_result = cv2.addWeighted(binary_colored, 1.0, highlight_b_array, 1.0, 0)
            output_d_path = os.path.join(show_dir, f'D_processed_{F}.png')
            cv2.imwrite(output_d_path, overlay_result)
            print(f"✅ D 结果已保存: {output_d_path}")
    
            # 【新增】生成 D_high 图：在过滤后的 B 图中，将所有前景点高亮为绿色，再叠加 C 图前景点（直接覆盖）
            D_high = np.zeros_like(filtered_B_image, dtype=np.uint8)
            # 对于非零像素，设置为绿色
            green_channel = (filtered_B_image > 0).astype(np.uint8)*255
            D_high = cv2.merge([np.zeros_like(green_channel), green_channel, np.zeros_like(green_channel)])
            # 叠加 C 图前景点（直接覆盖）
            for pt in C_points:
                pt_xy = tuple(pt[::-1].astype(int))
                cv2.circle(D_high, pt_xy, 1, (255,255,255), -1)
            output_D_high = os.path.join(show_dir, f'D_high_{F}.png')
            cv2.imwrite(output_D_high, D_high)
            print(f"✅ D_high 文件已保存: {output_D_high}")
    
            # 4. 读取 D 图（刚保存的叠加结果）与 E 图（tree部分处理后的结果）
            output_e_path = os.path.join(tree_dir, f'Filtered_E{F}_{R}.png')
            imgD = cv2.imread(output_d_path, cv2.IMREAD_COLOR)
            imgE = cv2.imread(output_e_path, cv2.IMREAD_COLOR)
            if imgD is None or imgE is None:
                print(f"⚠ D 或 E 图加载失败: {output_d_path} 或 {output_e_path}")
                continue
            # 转换为灰度图后提取前景点
            grayD = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)
            grayE = cv2.cvtColor(imgE, cv2.COLOR_BGR2GRAY)
            D_points = np.column_stack(np.where(grayD > 0))
            E_points = np.column_stack(np.where(grayE > 0))
            if D_points.shape[0] == 0 or E_points.shape[0] == 0:
                print(f"⚠ D 或 E 图未检测到前景点: F={F}")
                continue
    
            # 5. 计算 D′（去除与 C 图前景点相交的点）和 E′，并取交集 X = D′ ∩ E′
            C_set = set(map(tuple, C_points))
            D_prime = np.array([pt for pt in D_points if tuple(pt) not in C_set])
            E_prime = np.array([pt for pt in E_points if tuple(pt) not in C_set])
            D_prime_set = set(map(tuple, D_prime))
            E_prime_set = set(map(tuple, E_prime))
            X_set = D_prime_set & E_prime_set
            X = np.array(list(X_set))
            
            # 6. 读取 A 图（彩色图），转换为灰度图后提取前景点，并利用 IQR 方法去除异常值，
            #    再求与 C 图前景点的交集 Y
            image_a_path = os.path.join(deal_path, f'{F}%_c1.tif')
            imgA = cv2.imread(image_a_path, cv2.IMREAD_COLOR)
            if imgA is None:
                print(f"⚠ A 图加载失败: {image_a_path}")
                continue
            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            A_indices = np.column_stack(np.where(grayA > 0))
            A_values = grayA[grayA > 0]
            if A_values.size == 0:
                print(f"⚠ A 图未检测到前景点: F={F}")
                continue
            valid_indices_A = remove_outlier_indices(A_values, m)
            # 异常点即为未被保留的点
            all_indices_A = np.arange(A_values.size)
            outlier_indices_A = np.setdiff1d(all_indices_A, valid_indices_A)
            A_filtered_points = A_indices[valid_indices_A]
            A_filtered_set = set(map(tuple, A_filtered_points))
            Y_set = A_filtered_set & C_set
            Y = np.array(list(Y_set))
            
            # 【新增】在原 A 图上，将被过滤的异常点标记为黄色（BGR：(0,255,255)）
            A_image_color = imgA.copy()
            outlier_A_coords = A_indices[outlier_indices_A]
            for pt in outlier_A_coords:
                pt_xy = tuple(pt[::-1].astype(int))
                cv2.circle(A_image_color, pt_xy, 1, (0,255,255), -1)
            output_A_filter = os.path.join(show_dir, f'A_filter_{F}.png')
            cv2.imwrite(output_A_filter, A_image_color)
            print(f"✅ A_filter 文件已保存: {output_A_filter}")
    
            print(f"F={F}% 图像中 X 点集大小: {len(X)}")
            
            # 7. 计算 X 点集对应的荧光表达量（S_X）：利用过滤后的 B 图求平均表达量
            if len(X_set) > 0:
                avg_expression = np.mean([filtered_B_image[pt[0], pt[1]] for pt in X_set])
            else:
                avg_expression = 0
            print(f"X_set 中所有点的荧光表达量平均值 avg_expression = {avg_expression}")
            
            # 8. 计算结构效率和基线活动比：
            #    结构效率 = |X| / |E′|
            #    基线活动比 = |Y| / |C|
            if len(E_prime) > 0 and len(C_set) > 0:
                structural_efficiency = len(X) / len(E_prime)
                baseline_activity_ratio = len(Y) / len(C_set)
            else:
                structural_efficiency = 0
                baseline_activity_ratio = 0
                
            # 定义 SIF（结构指标）：结构效率 × 基线活动比 × 平均荧光表达量
            Z = structural_efficiency * baseline_activity_ratio * avg_expression
            Z_values.append(Z)
            struct_eff_list.append(structural_efficiency)
            base_act_list.append(baseline_activity_ratio)
            avg_expr_list.append(avg_expression)
            
            # # 【新增】生成 A_high 图：在过滤后的 A 图上，将所有前景点高亮为紫色（BGR：(255,0,255)）
            # # 先复制过滤后的 A 图（只保留有效点，其他置0）
            # A_filtered_image = np.zeros_like(grayA)
            # A_filtered_image[tuple(A_filtered_points.T)] = grayA[tuple(A_filtered_points.T)]
            # # 转换为彩色图
            # A_filtered_color = cv2.cvtColor(A_filtered_image, cv2.COLOR_GRAY2BGR)
            # # 将所有前景点高亮为紫色
            # A_high = np.where(A_filtered_image[..., None] > 0, (255,0,255), A_filtered_color)
            # A_high = A_high.astype(np.uint8)
            # # 叠加 C 图前景点，以半透明形式（alpha=0.5）
            # C_overlay = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
            # overlay_A_high = cv2.addWeighted(A_high, 0.5, C_overlay, 0.5, 0)
            # output_A_high = os.path.join(show_dir, f'A_high_{F}.png')
            # cv2.imwrite(output_A_high, overlay_A_high)
            # print(f"✅ A_high 文件已保存: {output_A_high}")
            
            
            # A_C=A_filtered_color.astype(np.uint8)
            # overlay_A_C = cv2.addWeighted(A_C, 0.5, C_overlay, 0.5, 0)
            # output_A_C = os.path.join(show_dir, f'A_C_{F}.png')
            # cv2.imwrite(output_A_C, overlay_A_C)
            # print(f"✅ A_C 文件已保存: {output_A_C}")
            
            
            
            
            # 【新增】生成 A_high 图：在过滤后的 A 图上，将所有前景点高亮为紫色（BGR：(255,0,255)）
            # 先复制过滤后的 A 图（只保留有效点，其他置0）
            A_filtered_image = np.zeros_like(grayA)
            A_filtered_image[tuple(A_filtered_points.T)] = grayA[tuple(A_filtered_points.T)]

            # 转换为彩色图
            A_filtered_color = cv2.cvtColor(A_filtered_image, cv2.COLOR_GRAY2BGR)

            # 将所有前景点高亮为紫色
            A_high = np.where(A_filtered_image[..., None] > 0, (255, 0, 255), A_filtered_color)
            A_high = A_high.astype(np.uint8)

            # 叠加 C 图前景点，以半透明形式（alpha=0.5）
            C_overlay = cv2.cvtColor(binaryC, cv2.COLOR_GRAY2BGR)
            overlay_A_high = cv2.addWeighted(A_high, 0.5, C_overlay, 0.5, 0)
            output_A_high = os.path.join(show_dir, f'A_high_{F}.png')
            cv2.imwrite(output_A_high, overlay_A_high)
            print(f"✅ A_high 文件已保存: {output_A_high}")

            # 生成 A_C 图：保持原色
            # 直接使用原始彩色图像中的有效点
            A_C = np.zeros_like(imgA)  # 使用原始彩色图像 imgA 的形状
            A_C[tuple(A_filtered_points.T)] = imgA[tuple(A_filtered_points.T)]

            # 叠加 C 图前景点，以半透明形式（alpha=0.5）
            overlay_A_C = cv2.addWeighted(A_C, 0.5, C_overlay, 0.5, 0)
            output_A_C = os.path.join(show_dir, f'A_C_{F}_2.png')
            cv2.imwrite(output_A_C, overlay_A_C)
            print(f"✅ A_C 文件已保存: {output_A_C}")
            

            
            
    
            # 9. 保存当前 F 的计算结果，同时记录 R、SIF、structural_efficiency、baseline_activity_ratio、avg_expression
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

        #################################################优化
        try:
            A_np = np.array(A, dtype=float)
            is_numeric = True
        except:
            is_numeric = False
    
        if is_numeric and len(Z_values) == len(A):
            # 数值型时，执行原来的线性回归拟合流程
            Z_values_np = np.array(Z_values)
            # 线性回归拟合：Z作为自变量，A作为因变量
            slope, intercept, r_value, p_value, _ = stats.linregress(Z_values_np, A_np)
            # 添加回归结果记录
            all_results.append({
                "F": "N/A",
                "R": R,
                "Fit": f"Y = {slope:.4f} * Z + {intercept:.4f}",
                "r_value": round(r_value, 4),
                "p_value": round(p_value, 4)
            })
            # 绘制拟合图
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
            print(f"✅ 拟合图已保存: {fit_file}")
        else:
            # 非数值型：绘制折线图，A为横坐标，Z_values 为纵坐标（标记为“SIF”）
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
            print(f"✅ 折线图已保存: {line_file}")        
        # 绘制 structural_efficiency 与 baseline_activity_ratio 的折线图
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
        print(f"✅ Structural Efficiency 和 Baseline Activity Ratio 折线图已保存: {lineplot1_file}")
    
        # 绘制 avg_expression 的折线图（单独绘制）
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
        print(f"✅ Average Expression 折线图已保存: {lineplot2_file}")
    
        # 将当前R的所有结果保存到一个DataFrame中，并写入综合Excel的一个sheet
        df_results = pd.DataFrame(all_results)
        # 将sheet名称设置为 "R_当前R值"
        df_results.to_excel(writer_all, sheet_name=f'R_{R}', index=False)
        print(f"\n✅ 当前R={R}的结果已写入综合Excel的Sheet: R_{R}")
    
# 退出with块时，ExcelWriter会自动保存文件
print(f"\n✅ 所有R的综合结果已保存到 {excel_show_all_path}")    
