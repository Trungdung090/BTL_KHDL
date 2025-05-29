from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import json
import os
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import threading

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dung'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Tạo thư mục upload (nếu chưa có)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class StockPredictor:
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        }

    def load_data(self, file_path):
        """Tải dữ liệu từ file CSV"""
        try:
            self.data = pd.read_csv(file_path, low_memory=False)
            if self.data.empty:
                print("Dữ liệu rỗng")
                return False

            # Chuẩn hóa tên cột
            column_mapping = {
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            # Đổi tên cột (nếu cần)
            for old_col, new_col in column_mapping.items():
                if old_col in self.data.columns:
                    self.data.rename(columns={old_col: new_col}, inplace=True)

            # Chuyển đổi cột date
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
                # Remove rows with invalid dates
                self.data = self.data.dropna(subset=['date'])
                self.data = self.data.sort_values('date')

            # Thêm cột symbol nếu chưa có
            if 'symbol' not in self.data.columns:
                self.data['symbol'] = 'STOCK'

            # Validate required columns
            required_columns = ['close', 'volume', 'high', 'low']
            for col in required_columns:
                if col not in self.data.columns:
                    print(f"Thiếu cột bắt buộc: {col}")
                    return False

            # Remove rows with NaN values in critical columns
            self.data = self.data.dropna(subset=required_columns)

            # Check if we have enough data after cleaning
            if len(self.data) < 100:
                print(f"Dữ liệu không đủ. Cần ít nhất 100 dòng, chỉ có {len(self.data)} dòng")
                return False
            print(f"Đã tải thành công {len(self.data)} dòng dữ liệu")
            return True
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return False

    def calculate_technical_indicators(self):
        """Tính các chỉ số kỹ thuật"""
        if self.data is None or len(self.data) < 50:
            return False
        try:
            # Simple Moving Average (SMA)
            self.data['sma_20'] = self.data['close'].rolling(window=20, min_periods=1).mean()
            self.data['sma_50'] = self.data['close'].rolling(window=50, min_periods=1).mean()

            # Exponential Moving Average (EMA)
            self.data['ema_20'] = self.data['close'].ewm(span=20, min_periods=1).mean()

            # Relative Strength Index (RSI)
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()

            # Avoid division by zero
            rs = gain / (loss + 1e-10)
            self.data['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            self.data['bb_middle'] = self.data['close'].rolling(window=20, min_periods=1).mean()
            bb_std = self.data['close'].rolling(window=20, min_periods=1).std()
            self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
            self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)

            # MACD
            exp1 = self.data['close'].ewm(span=12, min_periods=1).mean()
            exp2 = self.data['close'].ewm(span=26, min_periods=1).mean()
            self.data['macd'] = exp1 - exp2
            self.data['macd_signal'] = self.data['macd'].ewm(span=9, min_periods=1).mean()

            # Fill any remaining NaN values with forward fill then backward fill
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            return True
        except Exception as e:
            print(f"Lỗi khi tính chỉ số kỹ thuật: {e}")
            return False

    def prepare_features(self, days_back=30):
        """Chuẩn bị dữ liệu cho mô hình ML"""
        if self.data is None or len(self.data) < days_back + 50:
            print("Không đủ dữ liệu để chuẩn bị features")
            return None, None
        try:
            # Tính các chỉ số kỹ thuật
            if not self.calculate_technical_indicators():
                return None, None
            # Tạo features
            features = []
            targets = []
            for i in range(days_back, len(self.data)):
                try:
                    # Lấy dữ liệu của days_back ngày trước
                    feature_row = []
                    for j in range(days_back):
                        idx = i - days_back + j
                        if idx >= 0 and idx < len(self.data):
                            feature_row.extend([
                                float(self.data.iloc[idx]['close']),
                                float(self.data.iloc[idx]['volume']),
                                float(self.data.iloc[idx]['high']),
                                float(self.data.iloc[idx]['low'])
                            ])
                        else:
                            # Use last known values if index is out of bounds
                            feature_row.extend([
                                float(self.data.iloc[max(0, idx)]['close']),
                                float(self.data.iloc[max(0, idx)]['volume']),
                                float(self.data.iloc[max(0, idx)]['high']),
                                float(self.data.iloc[max(0, idx)]['low'])
                            ])
                    # Thêm các chỉ số kỹ thuật
                    prev_idx = max(0, i - 1)
                    feature_row.extend([
                        float(self.data.iloc[prev_idx].get('sma_20', self.data.iloc[prev_idx]['close'])),
                        float(self.data.iloc[prev_idx].get('ema_20', self.data.iloc[prev_idx]['close'])),
                        float(self.data.iloc[prev_idx].get('rsi', 50))
                    ])
                    features.append(feature_row)
                    targets.append(float(self.data.iloc[i]['close']))
                except Exception as e:
                    print(f"Lỗi tại index {i}: {e}")
                    continue
            if len(features) == 0:
                print("Không tạo được features")
                return None, None
            return np.array(features), np.array(targets)
        except Exception as e:
            print(f"Lỗi khi chuẩn bị features: {e}")
            return None, None

    def train_models(self):
        """Huấn luyện các mô hình"""
        try:
            X, y = self.prepare_features()
            if X is None or len(X) == 0:
                return {'error': 'Không thể chuẩn bị dữ liệu training'}

            # Chia dữ liệu train/test
            split_idx = max(1, int(len(X) * 0.8))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if len(X_train) == 0 or len(X_test) == 0:
                return {'error': 'Không đủ dữ liệu để chia train/test'}
            # Chuẩn hóa dữ liệu
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Huấn luyện các mô hình
            results = {}
            for name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    results[name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'rmse': float(np.sqrt(mse))
                    }
                except Exception as e:
                    results[name] = {'error': f'Lỗi training model {name}: {str(e)}'}
            return results
        except Exception as e:
            return {'error': f'Lỗi training: {str(e)}'}

    def predict_future(self, model_name='random_forest', days=30):
        """Dự báo giá trong tương lai"""
        if self.data is None or len(self.data) < 60:
            return None
        try:
            model = self.models.get(model_name)
            if model is None:
                return None
            # Lấy dữ liệu 30 ngày gần nhất
            lookback_days = min(30, len(self.data))
            recent_data = self.data.tail(lookback_days).copy()

            if len(recent_data) == 0:
                return None
            predictions = []
            current_data = recent_data.copy()
            for i in range(min(days, 90)):  # Limit to 90 days max
                try:
                    # Chuẩn bị feature cho ngày tiếp theo
                    feature_row = []
                    data_len = len(current_data)

                    # Use available data, pad if necessary
                    for j in range(30):
                        idx = data_len - 30 + j
                        if idx >= 0 and idx < data_len:
                            row_data = current_data.iloc[idx]
                            feature_row.extend([
                                float(row_data['close']),
                                float(row_data['volume']),
                                float(row_data['high']),
                                float(row_data['low'])
                            ])
                        else:
                            # Use first or last available data
                            fallback_idx = max(0, min(idx, data_len - 1))
                            if data_len > 0:
                                row_data = current_data.iloc[fallback_idx]
                                feature_row.extend([
                                    float(row_data['close']),
                                    float(row_data['volume']),
                                    float(row_data['high']),
                                    float(row_data['low'])
                                ])
                            else:
                                feature_row.extend([0, 0, 0, 0])

                    # Thêm chỉ số kỹ thuật (sử dụng giá trị cuối)
                    if data_len > 0:
                        last_row = current_data.iloc[-1]
                        feature_row.extend([
                            float(last_row.get('sma_20', last_row['close'])),
                            float(last_row.get('ema_20', last_row['close'])),
                            float(last_row.get('rsi', 50))
                        ])
                    else:
                        feature_row.extend([0, 0, 50])

                    # Chuẩn hóa và dự báo
                    if len(feature_row) == self.scaler.n_features_in_:
                        feature_scaled = self.scaler.transform([feature_row])
                        predicted_price = float(model.predict(feature_scaled)[0])
                    else:
                        # Fallback prediction
                        predicted_price = float(current_data.iloc[-1]['close'])

                    # Tạo ngày tiếp theo
                    if data_len > 0:
                        last_date = current_data.iloc[-1]['date']
                        if pd.isna(last_date):
                            last_date = datetime.now() - timedelta(days=1)
                        next_date = last_date + timedelta(days=1)
                    else:
                        next_date = datetime.now() + timedelta(days=i)

                    # Thêm dự báo vào kết quả
                    predictions.append({
                        'date': next_date.strftime('%Y-%m-%d'),
                        'predicted_price': predicted_price
                    })

                    # Cập nhật dữ liệu hiện tại
                    new_row = {
                        'date': next_date,
                        'close': predicted_price,
                        'volume': float(current_data.iloc[-1]['volume']) if data_len > 0 else 1000,
                        'high': predicted_price * 1.02,
                        'low': predicted_price * 0.98,
                        'sma_20': predicted_price,
                        'ema_20': predicted_price,
                        'rsi': 50
                    }
                    current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                except Exception as e:
                    print(f"Lỗi tại bước prediction {i}: {e}")
                    break
            return predictions
        except Exception as e:
            print(f"Lỗi predict_future: {e}")
            return None

    def get_statistics(self):
        """Tính toán thống kê"""
        if self.data is None or len(self.data) == 0:
            return None
        try:
            return {
                'total_days': int(len(self.data)),
                'avg_price': float(self.data['close'].mean()),
                'max_price': float(self.data['close'].max()),
                'min_price': float(self.data['close'].min()),
                'avg_volume': float(self.data['volume'].mean()),
                'price_change': float(self.data['close'].iloc[-1] - self.data['close'].iloc[0]),
                'price_change_percent': float(
                    ((self.data['close'].iloc[-1] - self.data['close'].iloc[0]) / self.data['close'].iloc[0]) * 100),
                'current_price': float(self.data['close'].iloc[-1])
            }
        except Exception as e:
            print(f"Lỗi get_statistics: {e}")
            return None

    def create_chart(self, predictions=None):
        """Tạo biểu đồ"""
        if self.data is None or len(self.data) == 0:
            return None
        try:
            # Lấy 100 điểm dữ liệu gần nhất
            chart_data_points = min(100, len(self.data))
            recent_data = self.data.tail(chart_data_points)

            fig = go.Figure()

            # Thêm dữ liệu giá thực tế
            fig.add_trace(go.Scatter(
                x=recent_data['date'],
                y=recent_data['close'],
                mode='lines',
                name='Giá Thực Tế',
                line=dict(color='#3B82F6', width=2),
                hoverinfo='x+y',
                hovertemplate='%{x|%d/%m/%Y}<br>Giá: %{y:.2f}<extra></extra>'
            ))

            # Thêm SMA (nếu có)
            if 'sma_20' in recent_data.columns and not recent_data['sma_20'].isna().all():
                fig.add_trace(go.Scatter(
                    x=recent_data['date'],
                    y=recent_data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#F59E0B', width=1, dash='dash'),
                    hoverinfo='x+y',
                    hovertemplate='%{x|%d/%m/%Y}<br>SMA 20: %{y:.2f}<extra></extra>'
                ))

            # Thêm dự báo (nếu có)
            if predictions and len(predictions) > 0:
                try:
                    pred_dates = [datetime.strptime(pred['date'], '%Y-%m-%d') for pred in predictions]
                    pred_prices = [pred['predicted_price'] for pred in predictions]
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=pred_prices,
                        mode='lines+markers',
                        name='Dự Báo',
                        line=dict(color='#10B981', width=2, dash='dot'),
                        marker=dict(size=4),
                        hoverinfo='x+y',
                        hovertemplate='%{x|%d/%m/%Y}<br>Dự báo: %{y:.2f}<extra></extra>'
                    ))
                except Exception as e:
                    print(f"Lỗi khi thêm predictions vào chart: {e}")

            fig.update_layout(
                title='Biểu Đồ Giá Cổ Phiếu',
                xaxis_title='Ngày',
                yaxis_title='Giá ($)',
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.8)",
                    font_size=14,
                    font_family="Arial"
                )
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Lỗi create_chart: {e}")
            return None

# Khởi tạo predictor
predictor = StockPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    if file and file.filename.lower().endswith('.csv'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Tải dữ liệu
            if predictor.load_data(filepath):
                # Huấn luyện mô hình
                training_results = predictor.train_models()
                # Lấy thống kê
                stats = predictor.get_statistics()
                # Tạo biểu đồ
                chart = predictor.create_chart()
                return jsonify({
                    'success': True,
                    'message': 'Tải dữ liệu thành công',
                    'stats': stats,
                    'chart': chart,
                    'training_results': training_results
                })
            else:
                return jsonify({ 'error': 'Lỗi khi xử lý dữ liệu - Kiểm tra format file CSV và đảm bảo có đủ dữ liệu (ít nhất 100 dòng)'})
        except Exception as e:
            return jsonify({'error': f'Lỗi: {str(e)}'})
    return jsonify({'error': 'Chỉ chấp nhận file CSV'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data.get('model', 'random_forest')
        days = int(data.get('days', 30))
        days = min(max(1, days), 90)  # Limit between 1 and 90 days

        # Tạo dự báo
        predictions = predictor.predict_future(model_name, days)
        if predictions:
            # Tạo biểu đồ với dự báo
            chart = predictor.create_chart(predictions)
            return jsonify({
                'success': True,
                'predictions': predictions,
                'chart': chart
            })
        else:
            return jsonify({'error': 'Không thể tạo dự báo - Kiểm tra dữ liệu và mô hình'})
    except Exception as e:
        return jsonify({'error': f'Lỗi dự báo: {str(e)}'})

@app.route('/stats')
def get_stats():
    stats = predictor.get_statistics()
    if stats:
        return jsonify(stats)
    else:
        return jsonify({'error': 'Chưa có dữ liệu'})

if __name__ == '__main__':
    app.run(debug=True, threaded=False)