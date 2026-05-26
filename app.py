#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碳排放预测系统
基于Flask的Web应用，集成中国地图可视化和SHAP分析
"""

from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
import os
import pandas as pd
import json
from datetime import datetime
import shap

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'carbon_emission_prediction_system_2024'

# 配置
class Config:
    # 模型和数据路径
    MODEL_PATH = "models/xgboost_model.pkl"
    SCALER_PATH = "data/scaler.pkl"
    FEATURES_PATH = "data/selected_features.xlsx"
    NORMALIZATION_PARAMS_PATH = "data/normalization_params.json"
    # 外部模型路径（2026.3.6版本）
    MODEL_PATH = "models/xgboost_model.pkl"
    SCALER_PATH = "data/scaler.pkl"
    FEATURES_PATH = "data/selected_features.xlsx"
    NORMALIZATION_PARAMS_PATH = "data/normalization_params.json"
    # 地图数据路径
    MAP_DATA_PATH = r"data/china_carbon_emission_data.json"
    # SHAP值数据路径
    SHAP_DATA_PATH = r"data/shap_values.json"

# 特征名称中英文映射
FEATURE_MAPPING = {
    'BN': '桥梁数量',
    'GDP': '地区生产总值(万元)',
    'AVPI': '工业增加值(亿元)',
    'AVTI': '第三产业增加值(万元)',
    'GDPPC': '人均地区生产总值(元)',
    'FV': '私人汽车拥有量(辆)',
    'PCO': '公路货运量(万吨)',
    'RP': '常住人口(万人)',
    'UL': '城镇化水平',
    'UGCA': '城市绿化覆盖面积(公顷)',
    'PRP': '公路客运量(万人)',
    'HM': '高速公路里程(公里)',
    'PV': '公共交通车辆(辆)',
    'PD': '人口密度(人/平方公里)',
    'C': '碳排放强度(吨/万元)',
    'IAV': '工业增加值(亿元)'
}

# 特征数值范围说明
FEATURE_RANGES = {
    'BN': {'name': '桥梁数量', 'min': 1, 'max': 3678, 'unit': '座'},
    'GDP': {'name': '地区生产总值', 'min': 539487, 'max': 323880000, 'unit': '万元'},
    'AVPI': {'name': '工业增加值', 'min': 13.81, 'max': 12409.13, 'unit': '亿元'},
    'AVTI': {'name': '第三产业增加值', 'min': 209320.956, 'max': 206112333, 'unit': '万元'},
    'GDPPC': {'name': '人均地区生产总值', 'min': 2957, 'max': 467749, 'unit': '元'},
    'FV': {'name': '私人汽车拥有量', 'min': 398, 'max': 88352, 'unit': '辆'},
    'PCO': {'name': '公路货运量', 'min': 5273, 'max': 6258200, 'unit': '万吨'},
    'RP': {'name': '常住人口', 'min': 73.9, 'max': 1035, 'unit': '万人'},
    'UL': {'name': '城镇化水平', 'min': 0.1916, 'max': 1.0, 'unit': ''},
    'UGCA': {'name': '城市绿化覆盖面积', 'min': 277, 'max': 158649, 'unit': '公顷'},
    'PRP': {'name': '公路客运量', 'min': 110.27, 'max': 158731, 'unit': '万人'},
    'HM': {'name': '高速公路里程', 'min': 716, 'max': 63881, 'unit': '公里'},
    'PV': {'name': '公共交通车辆', 'min': 146, 'max': 185011, 'unit': '辆'},
    'PD': {'name': '人口密度', 'min': 116, 'max': 2648, 'unit': '人/平方公里'},
    'C': {'name': '碳排放强度', 'min': 9.61, 'max': 1604.79, 'unit': '吨/万元'}
}

# 创建输出目录
def create_output_dirs():
    """创建必要的输出目录"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)

# 加载模型和相关文件
def load_model():
    """加载模型、特征名称和归一化参数"""
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载特征名称 - 优先使用模型训练时的特征名称
    # 从重要驱动因素文件中读取
    feature_names = []
    try:
        driver_factors_path = os.path.join(current_dir, r"../2026.3.6/step2_correlation_analysis/important_driver_factors.txt")
        if os.path.exists(driver_factors_path):
            with open(driver_factors_path, 'r', encoding='utf-8') as f:
                content = f.read()
            import re
            match = re.search(r'驱动因素列表 \(\d+个\):\n\[(.*?)\]', content, re.DOTALL)
            if match:
                factors_str = match.group(1)
                feature_names = [factor.strip().strip("'").strip('"') for factor in factors_str.split(',')]
                print(f"成功加载特征名称: {feature_names}")
    except Exception as e:
        print(f"加载特征名称失败: {str(e)}")
    
    # 如果没有加载到特征名称，使用默认值（与模型训练一致）
    if not feature_names:
        feature_names = ['FV', 'PCO', 'GDP', 'UL', 'BN', 'IAV', 'GDPPC']
        print(f"使用默认特征名称: {feature_names}")
    
    # 加载归一化参数
    scaler = None
    try:
        external_params_path = os.path.join(current_dir, r"../2026.3.6/step1_data_preprocessing/normalization_params.json")
        if os.path.exists(external_params_path):
            with open(external_params_path, 'r', encoding='utf-8') as f:
                normalization_params = json.load(f)
            
            # 创建归一化器
            class SimpleScaler:
                def __init__(self, params):
                    self.params = params
                
                def transform(self, X):
                    """对输入数据进行归一化"""
                    X_scaled = []
                    for sample in X:
                        scaled_sample = []
                        for i, feature in enumerate(feature_names):
                            if feature in self.params:
                                min_val = self.params[feature]['min']
                                max_val = self.params[feature]['max']
                                if max_val - min_val > 0:
                                    scaled = (sample[i] - min_val) / (max_val - min_val)
                                else:
                                    scaled = 0.0
                            else:
                                # 使用特征的合理范围进行归一化
                                if feature == 'FV':  # 私人汽车拥有量
                                    scaled = (sample[i] - 398) / (88352 - 398)
                                elif feature == 'PCO':  # 公路货运量
                                    scaled = (sample[i] - 5273) / (6258200 - 5273)
                                elif feature == 'GDP':  # 地区生产总值
                                    scaled = (sample[i] - 539487) / (323880000 - 539487)
                                elif feature == 'UL':  # 城镇化水平
                                    scaled = sample[i]  # 已经是0-1范围
                                elif feature == 'BN':  # 桥梁数量
                                    scaled = (sample[i] - 1) / (3678 - 1)
                                elif feature == 'IAV':  # 工业增加值
                                    scaled = (sample[i] - 13.81) / (12409.13 - 13.81)
                                elif feature == 'GDPPC':  # 人均地区生产总值
                                    scaled = (sample[i] - 2957) / (467749 - 2957)
                                else:
                                    scaled = sample[i] / 10000  # 默认归一化
                            scaled_sample.append(scaled)
                        X_scaled.append(scaled_sample)
                    return np.array(X_scaled)
            
            scaler = SimpleScaler(normalization_params)
    except Exception as e:
        pass
    
    # 加载模型（如果存在）
    model = None
    try:
        external_model_path = os.path.join(current_dir, Config.EXTERNAL_MODEL_PATH)
        if os.path.exists(external_model_path):
            with open(external_model_path, 'rb') as f:
                model = pickle.load(f)
            print("成功加载XGBoost模型")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
    
    return model, scaler, feature_names

# 加载历史排放数据
def load_historical_emission_data():
    """加载历史排放数据"""
    import pandas as pd
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "data", "1970～2023年中国各省市CO2总排放量(v2024_GHG).xlsx")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 省份名称映射（与ECharts地图匹配）
        province_map = {
            '北京市': '北京',
            '天津市': '天津',
            '河北省': '河北',
            '山西省': '山西',
            '内蒙古自治区': '内蒙古',
            '辽宁省': '辽宁',
            '吉林省': '吉林',
            '黑龙江省': '黑龙江',
            '上海市': '上海',
            '江苏省': '江苏',
            '浙江省': '浙江',
            '安徽省': '安徽',
            '福建省': '福建',
            '江西省': '江西',
            '山东省': '山东',
            '河南省': '河南',
            '湖北省': '湖北',
            '湖南省': '湖南',
            '广东省': '广东',
            '广西壮族自治区': '广西',
            '海南省': '海南',
            '重庆市': '重庆',
            '四川省': '四川',
            '贵州省': '贵州',
            '云南省': '云南',
            '西藏自治区': '西藏',
            '陕西省': '陕西',
            '甘肃省': '甘肃',
            '青海省': '青海',
            '宁夏回族自治区': '宁夏',
            '新疆维吾尔自治区': '新疆',
            '香港特别行政区': '香港',
            '澳门特别行政区': '澳门',
            '台湾省': '台湾'
        }
        
        # 转换数据格式（只加载2000-2023年的数据）
        historical_data = {}
        for _, row in df.iterrows():
            year = row['年份']
            # 只加载2000-2023年的数据
            if 2000 <= year <= 2023:
                province = row['省']
                # 映射省份名称
                normalized_province = province_map.get(province, province)
                year_str = str(year)
                emission = float(row['CO2排放量_吨']) / 10000  # 转换为万吨
                
                if normalized_province not in historical_data:
                    historical_data[normalized_province] = {}
                historical_data[normalized_province][year_str] = round(emission, 2)
        
        return historical_data
    except Exception as e:
        # 如果加载失败，返回空字典
        return {}

# 加载地图数据
def load_map_data():
    """加载地图数据"""
    # 首先尝试加载历史排放数据
    historical_data = load_historical_emission_data()
    if historical_data:
        return historical_data
    
    # 如果历史数据加载失败，尝试加载JSON数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_data_path = os.path.join(current_dir, Config.MAP_DATA_PATH)
    
    try:
        with open(map_data_path, 'r', encoding='utf-8') as f:
            map_data = json.load(f)
        return map_data
    except Exception as e:
        # 返回默认数据
        return generate_default_map_data()

# 生成默认地图数据
def generate_default_map_data():
    """生成默认地图数据"""
    provinces = [
        "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
        "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
        "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
        "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"
    ]
    
    # 生成2016-2025年的数据
    years = list(range(2016, 2026))
    data = {}
    
    for province in provinces:
        data[province] = {}
        for year in years:
            # 生成合理的碳排放数据（万吨）
            base_value = 1000 + (ord(province[0]) % 10) * 500
            trend = (year - 2016) * 100
            noise = np.random.randint(-100, 100)
            data[province][str(year)] = base_value + trend + noise
    
    return data

# 加载模型和数据
model, scaler, feature_names = load_model()
map_data = load_map_data()

# 首页
@app.route('/')
def home():
    """首页"""
    return render_template('index.html', feature_names=feature_names, map_data=map_data, 
                          feature_mapping=FEATURE_MAPPING, feature_ranges=FEATURE_RANGES)

# 历史预测记录接口
@app.route('/prediction_history', methods=['GET'])
def get_prediction_history():
    """获取历史预测记录"""
    history = session.get('prediction_history', [])
    return jsonify({
        'success': True,
        'history': history[-20:]  # 返回最近20条记录
    })

# 清空历史记录接口
@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空历史预测记录"""
    session['prediction_history'] = []
    return jsonify({'success': True, 'message': '历史记录已清空'})

# 预测接口
@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 获取输入数据
        data = request.json
        
        # 提取特征值
        features = []
        for feature in feature_names:
            value = float(data.get(feature, 0))
            features.append(value)
        
        # 转换为numpy数组
        features_array = np.array([features])
        
        # 数据归一化
        if scaler:
            scaled_features = scaler.transform(features_array)
        else:
            scaled_features = features_array
        
        # 预测
        if model:
            # 模型输出归一化值，需要反归一化
            normalized_prediction = float(model.predict(scaled_features)[0])
            
            # 获取归一化参数
            if scaler and hasattr(scaler, 'params') and 'C' in scaler.params:
                c_min = scaler.params['C']['min']
                c_max = scaler.params['C']['max']
                # 反归一化：实际值 = 归一化值 × (最大值 - 最小值) + 最小值
                prediction = normalized_prediction * (c_max - c_min) + c_min
            else:
                # 如果没有归一化参数，直接使用预测值
                prediction = normalized_prediction
        else:
            # 生成模拟预测结果
            prediction = float(np.random.uniform(50, 2000))
        
        # 使用真实的SHAP分析计算SHAP值
        shap_values = {}
        try:
            if model and scaler:
                # 加载归一化数据作为背景数据
                current_dir = os.path.dirname(os.path.abspath(__file__))
                normalized_data_path = os.path.join(current_dir, r"../2026.3.6/step1_data_preprocessing/normalized_data.xlsx")
                
                if os.path.exists(normalized_data_path):
                    # 读取归一化数据
                    normalized_df = pd.read_excel(normalized_data_path)
                    
                    # 确保使用与模型训练一致的特征顺序
                    model_feature_order = ['FV', 'PCO', 'GDP', 'UL', 'BN', 'IAV', 'GDPPC']
                    
                    # 选择特征并按正确顺序排列
                    X_background = normalized_df[model_feature_order].values
                    
                    # 创建SHAP解释器
                    explainer = shap.Explainer(model, X_background)
                    
                    # 对输入数据进行归一化
                    ordered_features = [features[feature_names.index(feature)] if feature in feature_names else 0 for feature in model_feature_order]
                    scaled_features = scaler.transform([ordered_features])
                    
                    # 计算SHAP值
                    shap_values_array = explainer(scaled_features)
                    
                    # 提取SHAP值，将负的SHAP值转为正值（因为我们要显示对总碳排放量的贡献）
                    for i, feature in enumerate(model_feature_order):
                        shap_value = float(abs(shap_values_array.values[0][i]))
                        shap_values[feature] = round(shap_value, 6)
                    print("成功计算真实SHAP值")
                else:
                    # 如果没有训练数据，使用基于shap_analysis.py的重要性计算
                    shap_weights = {
                        'BN': 0.0252,
                        'GDP': 0.0214,
                        'FV': 0.0148,
                        'PCO': 0.0147,
                        'UL': 0.0136,
                        'IAV': 0.0063,
                        'GDPPC': 0.0051
                    }
                    for feature in shap_weights:
                        shap_values[feature] = shap_weights[feature]
            else:
                # 模型未加载，使用固定的重要性
                shap_weights = {
                    'BN': 0.0252,
                    'GDP': 0.0214,
                    'FV': 0.0148,
                    'PCO': 0.0147,
                    'UL': 0.0136,
                    'IAV': 0.0063,
                    'GDPPC': 0.0051
                }
                for feature in shap_weights:
                    shap_values[feature] = shap_weights[feature]
        except Exception as e:
            print(f"计算SHAP值失败: {str(e)}")
            # 失败时使用固定的SHAP值
            shap_weights = {
                'BN': 0.0252,
                'GDP': 0.0214,
                'FV': 0.0148,
                'PCO': 0.0147,
                'UL': 0.0136,
                'IAV': 0.0063,
                'GDPPC': 0.0051
            }
            for feature in shap_weights:
                shap_values[feature] = shap_weights[feature]
        
        # 保存到历史记录
        history_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': round(prediction, 2),
            'features': dict(zip(feature_names, features)),
            'shap_values': shap_values
        }
        
        # 获取现有历史记录
        history = session.get('prediction_history', [])
        history.append(history_record)
        # 只保留最近50条记录
        if len(history) > 50:
            history = history[-50:]
        session['prediction_history'] = history
        
        # 格式化结果
        result = {
            'success': True,
            'prediction': round(prediction, 2),
            'features': dict(zip(feature_names, features)),
            'shap_values': shap_values
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# 健康检查
@app.route('/health')
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'feature_count': len(feature_names)
    })

# 地图数据接口
@app.route('/map_data')
def get_map_data():
    """地图数据接口"""
    return jsonify({
        'success': True,
        'data': map_data
    })

# 模型更新接口
@app.route('/update_model', methods=['POST'])
def update_model():
    """模型更新接口"""
    try:
        # 重新加载模型和数据
        global model, scaler, feature_names
        model, scaler, feature_names = load_model()
        
        # 重新加载地图数据
        global map_data
        map_data = load_map_data()
        
        return jsonify({
            'success': True,
            'message': '模型更新成功',
            'model_loaded': model is not None,
            'feature_count': len(feature_names),
            'features': feature_names
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# 深度分析接口
@app.route('/deep_analysis', methods=['GET'])
def deep_analysis():
    """深度分析接口"""
    try:
        # 加载SHAP分析结果
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shap_results_path = os.path.join(current_dir, r"../2026.3.6/step4_shap_analysis/results")
        
        # 检查结果目录是否存在
        if not os.path.exists(shap_results_path):
            # 运行SHAP分析脚本
            import subprocess
            shap_script_path = os.path.join(current_dir, r"../2026.3.6/step4_shap_analysis/shap_analysis.py")
            subprocess.run(['python', shap_script_path], cwd=os.path.dirname(shap_script_path))
        
        # 加载特征重要性数据
        feature_importance_path = os.path.join(shap_results_path, 'shap_feature_importance.xlsx')
        feature_importance_df = pd.read_excel(feature_importance_path)
        
        # 转换为字典格式
        feature_importance = []
        for _, row in feature_importance_df.iterrows():
            feature_importance.append({
                'feature': row['Feature'],
                'importance': float(row['SHAP Importance'])
            })
        
        # 加载分析报告
        report_path = os.path.join(shap_results_path, 'shap_analysis_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = f.read()
        else:
            report = "分析报告生成中..."
        
        return jsonify({
            'success': True,
            'feature_importance': feature_importance,
            'report': report,
            'features': feature_names
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# 省份详情页面
@app.route('/province/<province_name>')
def province_detail(province_name):
    """省份详情页面"""
    try:
        # 加载地图数据
        historical_data = load_historical_emission_data()
        
        # 获取省份数据
        province_data = {}
        max_emission = 0
        min_emission = float('inf')
        total_emission = 0
        count = 0
        max_year = ''
        min_year = ''
        
        # 检查省份是否存在
        if province_name in historical_data:
            province_data = historical_data[province_name]
            
            # 计算统计信息
            for year, emission in province_data.items():
                if emission > max_emission:
                    max_emission = emission
                    max_year = year
                if emission < min_emission:
                    min_emission = emission
                    min_year = year
                total_emission += emission
                count += 1
            
            # 计算平均值
            avg_emission = total_emission / count if count > 0 else 0
        else:
            # 省份不存在，使用默认值
            max_emission = 0
            min_emission = 0
            avg_emission = 0
            max_year = '2023'
            min_year = '2000'
        
        return render_template('province_detail.html', 
                           province=province_name, 
                           province_data=province_data, 
                           max_emission=max_emission, 
                           min_emission=min_emission, 
                           avg_emission=avg_emission, 
                           max_year=max_year, 
                           min_year=min_year)
    except Exception as e:
        return render_template('province_detail.html', 
                           province=province_name, 
                           province_data={}, 
                           max_emission=0, 
                           min_emission=0, 
                           avg_emission=0, 
                           max_year='2023', 
                           min_year='2000')

# 省份地图页面
@app.route('/province_map/<province_name>')
def province_map(province_name):
    """省份地图页面"""
    try:
        # 加载地图数据
        historical_data = load_historical_emission_data()
        
        # 获取省份数据
        province_data = {}
        max_emission = 0
        min_emission = float('inf')
        total_emission = 0
        count = 0
        max_year = ''
        min_year = ''
        
        # 检查省份是否存在
        if province_name in historical_data:
            province_data = historical_data[province_name]
            
            # 计算统计信息
            for year, emission in province_data.items():
                if emission > max_emission:
                    max_emission = emission
                    max_year = year
                if emission < min_emission:
                    min_emission = emission
                    min_year = year
                total_emission += emission
                count += 1
            
            # 计算平均值
            avg_emission = total_emission / count if count > 0 else 0
        else:
            # 省份不存在，使用默认值
            max_emission = 0
            min_emission = 0
            avg_emission = 0
            max_year = '2023'
            min_year = '2000'
        
        return render_template('province_map.html', 
                           province=province_name, 
                           province_data=province_data, 
                           max_emission=max_emission, 
                           min_emission=min_emission, 
                           avg_emission=avg_emission, 
                           max_year=max_year, 
                           min_year=min_year)
    except Exception as e:
        return render_template('province_map.html', 
                           province=province_name, 
                           province_data={}, 
                           max_emission=0, 
                           min_emission=0, 
                           avg_emission=0, 
                           max_year='2023', 
                           min_year='2000')

if __name__ == '__main__':
    # 创建输出目录
    create_output_dirs()
    
    # 获取端口（Railway使用环境变量PORT）
    port = int(os.environ.get('PORT', 8081))
    
    # 启动应用
    app.run(debug=False, host='0.0.0.0', port=port)