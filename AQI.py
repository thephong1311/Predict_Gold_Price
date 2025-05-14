import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

st.title("🌫️ Nền tảng dự báo ô nhiễm không khí đa chỉ số tại làng nghề")
st.image("0310khongkhi1.jpg", use_container_width=True)


# Chọn nguồn dữ liệu
option = st.radio("📍 Chọn làng nghề để dự báo:", ["Phú Vinh, Chương Mỹ, Hà Nội", "Làng nghề khác (tải lên dữ liệu)"])

if option == "Phú Vinh, Chương Mỹ, Hà Nội":
    df = pd.read_csv("air_quality_hourly_data.csv")  # Đảm bảo file này có trong repo khi deploy
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    st.success("📂 Đã nạp dữ liệu mặc định cho làng nghề Phú Vinh")

elif option == "Làng nghề khác (tải lên dữ liệu)":
    uploaded_file = st.file_uploader("📄 Tải lên file dữ liệu (.csv)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        st.success("📂 Đã tải thành công dữ liệu bạn cung cấp")
    else:
        st.warning("📎 Vui lòng tải lên file .csv để tiếp tục")
        st.stop()

    st.subheader("📌 Thông tin dữ liệu bạn đã tải lên:")
    st.write(df.head())

    st.subheader("📊 Mô tả thống kê dữ liệu:")
    st.write(df.describe())

    # Cho phép người dùng chọn biến mục tiêu (đầu ra) và biến đi kèm (đầu vào đi kèm)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Chọn biến mục tiêu
    st.subheader("🎯 Lựa chọn biến cho mô hình dự báo")
    target_variable = st.selectbox(
        "Chọn biến mục tiêu (Đầu ra):",
        options=numeric_columns,
        index=numeric_columns.index("AQI") if "AQI" in numeric_columns else 0
    )
    
    # Chọn biến đi kèm
    # Loại bỏ biến mục tiêu khỏi danh sách các biến đi kèm có thể chọn
    input_variables = [col for col in numeric_columns if col != target_variable]
    selected_input = st.multiselect(
        "Chọn biến đi kèm (Đầu vào):",
        options=input_variables,
        default=[input_variables[0]] if input_variables else []
    )
    
    # Đảm bảo có ít nhất một biến đi kèm
    if not selected_input:
        st.warning("⚠️ Vui lòng chọn ít nhất một biến đi kèm để dự báo.")
    else:
        # Vẽ biểu đồ biến mục tiêu theo thời gian
        st.subheader(f"📈 Diễn biến {target_variable} theo thời gian:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Timestamp'], df[target_variable], label=target_variable, color='red', linewidth=0.8)
        ax.set_xlabel("Thời gian", labelpad=10)
        ax.set_ylabel(target_variable, labelpad=10)
        ax.set_title(f"Biến động {target_variable} theo thời gian", pad=20)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout(pad=2)
        st.pyplot(fig)

        # Dự báo
        if st.button("🚀 Thực hiện dự báo"):
            # Chuẩn bị dữ liệu
            input_features = selected_input + [target_variable]
            output_feature = [target_variable]

            X = df[input_features].values
            y = df[output_feature].values

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y).flatten()

            split_index = int(len(df) * 0.8)
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

            def sin_cos_expansion(X):
                return np.hstack([X, np.sin(X), np.cos(X)])

            X_train_exp = sin_cos_expansion(X_train)
            X_test_exp = sin_cos_expansion(X_test)

            def train_flann(X_train_exp, y_train, learning_rate, epochs, lambda_reg):
                m, n = X_train_exp.shape
                weights = np.random.randn(n)
                for epoch in range(epochs):
                    y_pred = np.dot(X_train_exp, weights)
                    error = y_pred - y_train
                    gradient = (1 / m) * np.dot(X_train_exp.T, error) + lambda_reg * weights
                    weights -= learning_rate * gradient
                return weights

            weights = train_flann(X_train_exp, y_train, 0.05, 300, 0.1)

            y_test_pred = np.dot(X_test_exp, weights)
            y_test_pred_original = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

            timestamps = df['Timestamp'][split_index:].reset_index(drop=True)

            st.subheader(f"📤 Kết quả dự báo {target_variable} (giai đoạn kiểm tra):")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(timestamps, y_test_original.flatten(), label=f'Thực tế {target_variable}', color='orange')
            ax2.plot(timestamps, y_test_pred_original.flatten(), label=f'Dự báo {target_variable}', color='red', linestyle='--')
            ax2.set_title(f'📉 Dự báo {target_variable} giai đoạn kiểm tra', pad=15)
            ax2.set_xlabel("Thời gian")
            ax2.set_ylabel(target_variable)
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.7)
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45, ha='right')
            fig2.tight_layout()
            st.pyplot(fig2)

            # Chỉ hiển thị metric MAE theo yêu cầu
            mae = np.mean(np.abs(y_test_original.flatten() - y_test_pred_original.flatten()))
            st.metric("📉 MAE (Mean Absolute Error)", f"{mae:.2f}")

            result_df = pd.DataFrame({
                "Timestamp": timestamps,
                f"{target_variable}_ThucTe": y_test_original.flatten(),
                f"{target_variable}_DuBao": y_test_pred_original.flatten()
            })
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Tải kết quả dự báo", data=csv, file_name=f'du_bao_{target_variable}.csv', mime='text/csv')

        # Dự báo tương lai
        st.subheader(f"🔮 Dự báo {target_variable} cho những ngày tới")
        n_days_forecast = st.slider("Chọn số ngày muốn dự báo:", min_value=1, max_value=7, value=3)

        # Khởi tạo các biến nếu chưa được khởi tạo
        if st.button("📈 Dự báo tương lai"):
            # Đảm bảo người dùng đã chọn biến đi kèm
            if not selected_input:
                st.error("⚠️ Vui lòng chọn ít nhất một biến đi kèm để dự báo.")
            else:
                # Chuẩn bị dữ liệu và huấn luyện mô hình
                input_features = selected_input + [target_variable]
                output_feature = [target_variable]
                
                X = df[input_features].values
                y = df[output_feature].values
                
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y).flatten()
                
                def sin_cos_expansion(X):
                    return np.hstack([X, np.sin(X), np.cos(X)])
                    
                # Học mô hình từ toàn bộ dữ liệu
                X_exp = sin_cos_expansion(X_scaled)
                
                def train_flann(X_train_exp, y_train, learning_rate, epochs, lambda_reg):
                    m, n = X_train_exp.shape
                    weights = np.random.randn(n)
                    for epoch in range(epochs):
                        y_pred = np.dot(X_train_exp, weights)
                        error = y_pred - y_train
                        gradient = (1 / m) * np.dot(X_train_exp.T, error) + lambda_reg * weights
                        weights -= learning_rate * gradient
                    return weights
                    
                weights = train_flann(X_exp, y_scaled, 0.05, 300, 0.1)

                # Lấy các giá trị cuối cùng để bắt đầu dự báo
                last_target_value = df[target_variable].iloc[-1]
                
                # Lấy các giá trị gần đây của các biến đi kèm để tính giá trị trung bình
                recent_values = {}
                for col in selected_input:
                    recent_values[col] = df[col].iloc[-7:].mean()
                
                # Khởi tạo mảng để lưu kết quả dự báo
                future_dates = pd.date_range(start=df['Timestamp'].iloc[-1] + pd.Timedelta(days=1), periods=n_days_forecast)
                future_predictions = np.zeros(n_days_forecast)
                
                # Quá trình dự báo chuỗi thời gian - mỗi ngày dựa trên dự báo của ngày trước đó
                current_target_value = last_target_value
                current_input_values = dict(recent_values)
                
                # Tính toán độ lệch chuẩn của biến mục tiêu từ dữ liệu lịch sử
                target_std = df[target_variable].std() * 0.1  # Sử dụng 10% độ lệch chuẩn
                
                # Phân tích xu hướng biến mục tiêu từ 7 ngày gần nhất
                recent_target = df[target_variable].iloc[-7:].values
                if len(recent_target) >= 2:
                    recent_trend = np.mean(np.diff(recent_target))
                else:
                    recent_trend = 0
                
                # Thêm các hệ số biến động cho các biến
                target_noise_factor = 0.15  # Biến động cho biến mục tiêu
                input_noise_factors = {col: 0.2 for col in selected_input}  # Biến động cho các biến đầu vào
                
                for i in range(n_days_forecast):
                    # Chuẩn bị đầu vào cho dự báo
                    input_row = [current_input_values[col] for col in selected_input] + [current_target_value]
                    input_data = np.array([input_row])
                    
                    # Chuẩn hóa dữ liệu đầu vào
                    input_scaled = scaler_X.transform(input_data)
                    input_expanded = sin_cos_expansion(input_scaled)
                    
                    # Dự báo giá trị mới
                    pred_scaled = np.dot(input_expanded, weights)
                    base_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                    
                    # Thêm biến động và xu hướng vào dự báo
                    # Biến động ngẫu nhiên + xu hướng gần đây + thêm biến động tăng dần theo thời gian
                    time_factor = (i + 1) / 2  # Tăng biến động theo thời gian
                    target_random = np.random.normal(0, target_std * time_factor)
                    trend_component = recent_trend * (i + 1) * 0.5  # Phóng đại xu hướng theo thời gian
                    
                    # Tổng hợp các thành phần để có dự báo cuối cùng
                    final_pred = base_pred + target_random + trend_component
                    
                    # Đảm bảo giá trị dự báo không âm (nếu cần)
                    if final_pred < 0 and target_variable in ["AQI", "PM10", "PM2.5"]:
                        final_pred = abs(final_pred)
                    
                    # Lưu dự báo
                    future_predictions[i] = final_pred
                    
                    # Cập nhật cho lần dự báo tiếp theo
                    current_target_value = final_pred
                    
                    # Thêm biến động cho các biến đầu vào đi kèm
                    for col in selected_input:
                        input_random = np.random.normal(0, input_noise_factors[col] * current_input_values[col] * (i + 1))
                        current_input_values[col] = current_input_values[col] + input_random
                        
                        # Đảm bảo giá trị không âm (nếu cần)
                        if current_input_values[col] < 0 and col in ["PM10", "PM2.5"]:
                            current_input_values[col] = recent_values[col] * 0.7

                # Hiển thị kết quả dự báo
                fig_future, ax_future = plt.subplots(figsize=(12, 6))
                ax_future.plot(future_dates, future_predictions, marker='o', linestyle='--', color='red')
                ax_future.set_title(f"Dự báo {target_variable} cho {n_days_forecast} ngày tới", pad=15)
                ax_future.set_xlabel("Ngày")
                ax_future.set_ylabel(target_variable)
                ax_future.grid(True, linestyle=':', alpha=0.6)
                plt.xticks(rotation=45)
                fig_future.tight_layout()
                st.pyplot(fig_future)

                # Hiển thị thông tin chi tiết cho từng ngày
                for i in range(n_days_forecast):
                    pred_value = future_predictions[i]
                    date_str = future_dates[i].strftime('%d/%m/%Y')
                    
                    # Phân loại AQI nếu biến mục tiêu là AQI
                    if target_variable == "AQI":
                        if pred_value <= 50:
                            st.success(f"[{date_str}] AQI {pred_value:.1f} - 🌱 Tốt: Không khí trong lành. Bạn có thể hoạt động ngoài trời bình thường.")
                        elif pred_value <= 100:
                            st.info(f"[{date_str}] AQI {pred_value:.1f} - 🟡 Vừa phải: Chấp nhận được, người nhạy cảm nên hạn chế hoạt động dài ngoài trời.")
                        elif pred_value <= 150:
                            st.warning(f"[{date_str}] AQI {pred_value:.1f} - 🟠 Không lành mạnh cho nhóm nhạy cảm: Người mắc bệnh hô hấp nên tránh hoạt động kéo dài ngoài trời.")
                        elif pred_value <= 200:
                            st.error(f"[{date_str}] AQI {pred_value:.1f} - 🔴 Không khỏe mạnh: Mọi người nên hạn chế hoạt động ngoài trời.")
                        elif pred_value <= 300:
                            st.error(f"[{date_str}] AQI {pred_value:.1f} - 🟣 Rất không tốt: Tránh mọi hoạt động gắng sức ngoài trời.")
                        else:
                            st.error(f"[{date_str}] AQI {pred_value:.1f} - ⚫ Nguy hiểm: Ở trong nhà, tránh hoàn toàn các hoạt động ngoài trời.")
                    else:
                        # Hiển thị thông tin dự báo cho các biến khác
                        st.info(f"[{date_str}] {target_variable}: {pred_value:.2f}")
                
                # Tạo bảng kết quả dự báo để người dùng có thể tải về
                future_df = pd.DataFrame({
                    "Ngày": future_dates,
                    f"{target_variable}_DuBao": future_predictions
                })
                future_csv = future_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải kết quả dự báo tương lai", data=future_csv, file_name=f'du_bao_tuong_lai_{target_variable}.csv', mime='text/csv')
