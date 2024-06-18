import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# 定义读取 .mot 文件的函数
def read_mot_file(file):
    # 跳过文件头部，直到找到 'endheader'
    for line in file:
        decoded_line = line.decode('utf-8')
        if 'endheader' in decoded_line:
            break
    
    # 使用 Pandas 读取文件内容
    data = pd.read_csv(file, sep='\s+')

    return data

def getScoreA(trunk_score, neck_score, leg_score, score_map_a):
    posture_score_a = np.ones(len(trunk_score))  # 初始化结果数组
    for n in range(len(trunk_score)):
        key = f'{int(trunk_score[n])}-{int(neck_score[n])}-{int(leg_score[n])}'
        if key in score_map_a:
            score = score_map_a[key]
        else:
            score = np.nan  # 如果没有找到匹配项，返回 NaN
        posture_score_a[n] = score
    return posture_score_a

# 定义 getScoreB 函数
def getScoreB(upper_arm_score, lower_arm_score, wrist_score, score_map_b):
    posture_score_b = np.ones(len(upper_arm_score))  # 初始化结果数组
    for n in range(len(upper_arm_score)):
        key = f'{int(upper_arm_score[n])}-{int(lower_arm_score[n])}-{int(wrist_score[n])}'
        if key in score_map_b:
            score = score_map_b[key]
        else:
            score = np.nan  # 如果没有找到匹配项，返回 NaN
        posture_score_b[n] = score
    return posture_score_b
    
# 定义 getScoreC 函数
def getScoreC(posture_score_a, posture_score_b, score_map_c):
    posture_score_c = np.ones(len(posture_score_a))  # 初始化结果数组
    for n in range(len(posture_score_a)):
        key = f'{int(posture_score_a[n])}-{int(posture_score_b[n])}'
        if key in score_map_c:
            score = score_map_c[key]
        else:
            score = np.nan  # 如果没有找到匹配项，返回 NaN
        posture_score_c[n] = score
    return posture_score_c

# POSTUREA
max_trunk_score = 5
max_neck_score = 3
max_leg_score = 4

posture_A_keys = []
posture_A_values = [
    1, 2, 3, 4, 1, 2, 3, 4, 3, 3, 5, 6,
    2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7,
    2, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8,
    3, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9,
    4, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 9,
]

for trunk in range(1, max_trunk_score + 1):
    for neck in range(1, max_neck_score + 1):
        for leg in range(1, max_leg_score + 1):
            key = f'{trunk}-{neck}-{leg}'
            posture_A_keys.append(key)

scoreMapA = dict(zip(posture_A_keys, posture_A_values))

# POSTUREB
max_upper_score = 6
max_lower_score = 2
max_wrist_score = 3

posture_B_keys = []
posture_B_values = [
    1, 2, 2, 1, 2, 3,
    1, 2, 3, 2, 3, 4,
    3, 4, 5, 4, 5, 5,
    4, 5, 5, 5, 6, 7,
    6, 7, 8, 7, 8, 8,
    7, 8, 8, 8, 9, 9,
]

# 生成所有可能的组合键
for upper in range(1, max_upper_score + 1):
    for lower in range(1, max_lower_score + 1):
        for wrist in range(1, max_wrist_score + 1):
            key = f'{upper}-{lower}-{wrist}'
            posture_B_keys.append(key)

# 创建映射表
scoreMapB = dict(zip(posture_B_keys, posture_B_values))

# POSTUREC
max_postureA = 12
max_postureB = 12

posture_C_keys = []
posture_C_values = [
    1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7,
    1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8,
    2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8,
    3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9,
    4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9,
    6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10,
    7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11,
    8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11,
    9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12,
    10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12,
    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
]

# 生成所有可能的组合键
for postureA in range(1, max_postureA + 1):
    for postureB in range(1, max_postureB + 1):
        key = f'{postureA}-{postureB}'
        posture_C_keys.append(key)

# 创建映射表
scoreMapC = dict(zip(posture_C_keys, posture_C_values))

# Streamlit 界面
st.title('REBA Analysis by Kinematic data')

# 文件上传组件
uploaded_file = st.file_uploader("Choose a .mot file", type="mot")

if uploaded_file is not None:
    # 读取文件内容
    mot_data = read_mot_file(uploaded_file)
    # 将每一列数据命名为一个单独的变量
    for column in mot_data.columns:
        locals()[column] = mot_data[column].values
    # 进行分析
    # 示例分析：计算每个关节得分
        
    # 初始化 Trunk_score 数组
    Trunk_score = np.zeros(len(time))
    
    # 计算 Trunk_score
    for i in range(len(time)):
        Trunk_flex_angle = lumbar_extension[i]
        Trunk_rot_angle = lumbar_rotation[i]
        Trunk_bend_angle = lumbar_bending[i]
    
        if -5 <= Trunk_flex_angle < 5:
            Trunk_score[i] = 1
        elif -20 <= Trunk_flex_angle < -5:
            Trunk_score[i] = 2
        elif 5 <= Trunk_flex_angle < 20:
            Trunk_score[i] = 2
        elif 20 <= Trunk_flex_angle < 60:
            Trunk_score[i] = 3
        elif Trunk_flex_angle < -20:
            Trunk_score[i] = 3
        elif Trunk_flex_angle >= 60:
            Trunk_score[i] = 4
    
        # Consider if the trunk is rotated
        if Trunk_rot_angle >= 10:
            Trunk_score[i] += 1
        elif Trunk_rot_angle <= -10:
            Trunk_score[i] += 1
    
        # Consider if the trunk is bent
        if Trunk_bend_angle >= 10:
            Trunk_score[i] += 1
        elif Trunk_bend_angle <= -10:
            Trunk_score[i] += 1
     
    
    Leg_score = np.ones(len(time))

    # 计算 Leg_score
    for i in range(len(time)):
        Knee_flex_angle = knee_angle_r[i]
    
        if 30 <= Knee_flex_angle < 60:
            Leg_score[i] += 1
        elif Knee_flex_angle >= 60:
            Leg_score[i] += 2
        elif Knee_flex_angle < 30:
            Leg_score[i] = Leg_score[i]
    
    Neck_score = np.ones(len(time))
    
    # 假设 Trunk_score、Neck_score 和 Leg_score 已经计算完毕
    Posture_Score_A = getScoreA(Trunk_score, Neck_score, Leg_score, scoreMapA)
      
    Upper_arm_score = np.zeros(len(time))

    # 计算 Upper_arm_score
    for i in range(len(arm_flex_r)):
        arm_flex_angle = arm_flex_r[i]
        arm_add_angle = arm_add_r[i]
        
        if -20 <= arm_flex_angle < 20:
            Upper_arm_score[i] = 1
        elif 20 <= arm_flex_angle < 45:
            Upper_arm_score[i] = 2
        elif arm_flex_angle < -20:
            Upper_arm_score[i] = 2
        elif 45 <= arm_flex_angle < 90:
            Upper_arm_score[i] = 3
        elif arm_flex_angle >= 90:
            Upper_arm_score[i] = 4
        
        # Additional score for upper arm score
        if arm_add_angle <= -10:
            Upper_arm_score[i] += 1
    
    # 初始化 Lower_arm_score 数组
    Lower_arm_score = np.zeros(len(time))
    
    # 计算 Lower_arm_score
    for i in range(len(elbow_flex_r)):
        elbow_flex_angle = elbow_flex_r[i]
    
        if 60 <= elbow_flex_angle < 100:
            Lower_arm_score[i] = 1
        elif elbow_flex_angle < 60:
            Lower_arm_score[i] = 2
        elif elbow_flex_angle >= 100:
            Lower_arm_score[i] = 2
    
    # 在此研究中，腕部处于中立位置，没有显示出偏离中立位置的情况。
    Wrist_score = np.ones(len(time))
    
    # 计算 Posture_Score_B
    Posture_Score_B = getScoreB(Upper_arm_score, Lower_arm_score, Wrist_score, scoreMapB)
    
    Posture_Score_C = getScoreC (Posture_Score_A, Posture_Score_B, scoreMapC)
    
    # 查找最大值
    max_value = np.nanmax(Posture_Score_C)
    
    # 找到所有最大值的索引
    max_indices = np.where(Posture_Score_C == max_value)[0]
    
    # Import required libraries
        
    # 绘制 Posture_Score_C 图表
    st.write("Data analysis complete")
    
    # Assuming Posture_Score_C and time are already defined
    plt.figure(figsize=(10, 5))
    plt.plot(time, Posture_Score_C, marker='o', linestyle='-', color='b', label='REBA SCORE')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('REBA Score Over Time')
    plt.legend()
    plt.grid(True)
    
    # Set y-ticks to display integer values from 0 to 12
    plt.yticks(range(0, 13, 1))
    
    st.pyplot(plt)
    
    # 查找最大值
    max_value = np.nanmax(Posture_Score_C)
    
    # 找到所有最大值的索引
    max_indices = np.where(Posture_Score_C == max_value)[0]
    
    # 生成 DataFrame 展示这些最大值对应的 time，Posture_Score_C，Posture_Score_A，Posture_Score_B，Neck_score，Trunk_score，Leg_score，Upper_arm_score，Lower_arm_score，Wrist_score
    max_values_data = {
        'Time(s)': time[max_indices],
        'Posture_Score_C': Posture_Score_C[max_indices],
        'Posture_Score_A': Posture_Score_A[max_indices],
        'Posture_Score_B': Posture_Score_B[max_indices],
        'Neck_score': Neck_score[max_indices],
        'Trunk_score': Trunk_score[max_indices],
        'Leg_score': Leg_score[max_indices],
        'Upper_arm_score': Upper_arm_score[max_indices],
        'Lower_arm_score': Lower_arm_score[max_indices],
        'Wrist_score': Wrist_score[max_indices]
        
    }
    
    max_values_df = pd.DataFrame(max_values_data)
    
    # 在 Streamlit 页面上显示 DataFrame
    st.title("Maximum REBA Score Values and Corresponding Information")
    st.dataframe(max_values_df)
   
    
    
    
    
    
    