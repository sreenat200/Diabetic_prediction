import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import io

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load the dataset from GitHub URL
@st.cache_data
def load_data():
    try:
        # Define column names for the diabetes dataset
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                   'DiabetesPedigreeFunction', 'Age', 'Outcome']
        # Load from raw GitHub URL, treating first row as header
        url = "https://raw.githubusercontent.com/sreenat200/Diabetic_prediction/0e46abc455417092c9776719b322e5a13fc871f4/diabetes.csv"
        data = pd.read_csv(url, names=columns, delimiter=',', header=0, skipinitialspace=True, on_bad_lines='skip')
        # Ensure numeric data
        for col in columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        return data
    except Exception as e:
        st.error(f"Error loading file from GitHub: {e}")
        # Fallback: Allow user to upload a local CSV
        st.write("Please upload a local diabetes.csv file as a fallback.")
        uploaded_file = st.file_uploader("Upload diabetes.csv", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file, names=columns, delimiter=',', header=0, skipinitialspace=True, on_bad_lines='skip')
                for col in columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                return data
            except Exception as e2:
                st.error(f"Error loading uploaded file: {e2}")
        return None

# Preprocess the data
def preprocess_data(data, selected_features=None):
    if data is None:
        return None, None, None, None
    if selected_features is None:
        selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[selected_features]
    y = data['Outcome']
    
    # Handle missing values (replace zeros with median)
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if col in X.columns:
            X.loc[X[col] == 0, col] = X[col].replace(0, X[col].median())
    X = X.fillna(X.median())
    y = y.fillna(y.mode()[0])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

# Train the model
@st.cache_resource
def train_model(X, y, algorithm):
    if X is None or y is None:
        return None, None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif algorithm == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif algorithm == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if algorithm in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        feature_importance = model.feature_importances_
    else:
        feature_importance = np.abs(model.coef_[0]) / np.abs(model.coef_[0]).sum()
    
    return model, accuracy, precision, recall, f1, feature_importance, X_test, y_test, y_pred

# Generate PDF report
def generate_pdf_report(input_data, prediction, probability, accuracy, precision, recall, f1, feature_importance, features, algorithm):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph(f"Diabetes Prediction Report ({algorithm})", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Input Features", styles['Heading2']))
    data = [["Feature", "Value"]]
    for feature, value in input_data.items():
        data.append([feature, f"{value:.2f}" if isinstance(value, (int, float)) else str(value)])
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph(f"Prediction: {'Diabetic' if prediction else 'Non-Diabetic'}", styles['Heading2']))
    elements.append(Paragraph(f"Probability of Diabetes: {probability:.2%}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Model Performance", styles['Heading2']))
    elements.append(Paragraph(f"Algorithm: {algorithm}", styles['Normal']))
    elements.append(Paragraph(f"Accuracy: {accuracy:.4f}", styles['Normal']))
    elements.append(Paragraph(f"Precision: {precision:.4f}", styles['Normal']))
    elements.append(Paragraph(f"Recall: {recall:.4f}", styles['Normal']))
    elements.append(Paragraph(f"F1-Score: {f1:.4f}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Feature Importance", styles['Heading2']))
    importance_data = [["Feature", "Importance"]]
    for feature, importance in zip(features, feature_importance):
        importance_data.append([feature, f"{importance:.4f}"])
    importance_table = Table(importance_data)
    importance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(importance_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main app
def main():
    st.title("🩺 Diabetes Prediction App")
    st.markdown("Predict diabetes risk with live updates and customizable visualizations.")

    # Load and preprocess data
    data = load_data()
    if data is None:
        return

    # Sidebar for user input
    st.sidebar.header("🩺 Input Features")
    algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost"])
    
    input_data = {}
    input_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    for feature in input_features:
        if feature in ['Pregnancies', 'Age']:
            try:
                input_data[feature] = st.sidebar.slider(
                    feature,
                    min_value=int(data[feature].min()),
                    max_value=int(data[feature].max()),
                    value=int(data[feature].median()),
                    key=f"{feature}_input"
                )
            except ValueError as e:
                st.error(f"Error processing {feature}: {e}. Please check the dataset for non-numeric values.")
                return
        else:
            try:
                input_data[feature] = st.sidebar.number_input(
                    feature,
                    min_value=float(data[feature].min()),
                    max_value=float(data[feature].max()),
                    value=float(data[feature].median()),
                    step=0.1,
                    key=f"{feature}_input"
                )
            except ValueError as e:
                st.error(f"Error processing {feature}: {e}. Please check the dataset for non-numeric values.")
                return

    # Feature selection tab
    tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Selection", "Visualizations"])

    with tab2:
        st.header("Select Features for Training")
        selected_features = st.multiselect(
            "Choose features to train the model",
            options=input_features,
            default=input_features,
            key="feature_selection"
        )
        if not selected_features:
            st.warning("Please select at least one feature.")
            return
        
        # Train model with selected features
        X, y, scaler, feature_names = preprocess_data(data, selected_features)
        if X is None:
            return
        model, accuracy, precision, recall, f1, feature_importance, X_test, y_test, y_pred = train_model(X, y, algorithm)
        if model is None:
            return
        
        st.subheader("Model Performance with Selected Features")
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1-Score", f"{f1:.4f}")

    with tab1:
        st.header("Live Prediction")
        # Live prediction based on sidebar inputs
        input_df = pd.DataFrame([input_data])
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            if col in input_df.columns and input_df[col].iloc[0] == 0:
                input_df[col] = data[col].median()
        
        input_df = input_df[selected_features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.success(f"### Prediction: **{'Diabetic' if prediction == 1 else 'Non-Diabetic'}**")
        st.write(f"Probability of Diabetes: **{probability:.2%}**")

        pdf_buffer = generate_pdf_report(
            input_data,
            prediction,
            probability,
            accuracy,
            precision,
            recall,
            f1,
            feature_importance,
            feature_names,
            algorithm
        )
        st.download_button(
            label="📥 Download Prediction Report (PDF)",
            data=pdf_buffer,
            file_name=f"diabetes_prediction_{algorithm.replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.header("📈 Model Performance")
        st.metric("Algorithm", algorithm)
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1-Score", f"{f1:.4f}")

    with tab3:
        st.header("📊 Data Visualizations")
        
        # User selection for graph type
        graph_type = st.selectbox(
            "Select Visualization Type",
            options=["Confusion Matrix", "ROC Curve", "Feature Importance", "Scatter Plot"],
            key="graph_type"
        )
        
        if graph_type == "Scatter Plot":
            st.subheader("Select Features for Scatter Plot")
            x_feature = st.selectbox("X-axis Feature", options=input_features, key="x_feature")
            y_feature = st.selectbox("Y-axis Feature", options=input_features, key="y_feature")
            color_feature = st.selectbox("Color by", options=["Outcome"] + input_features, key="color_feature")
            size_feature = st.selectbox("Size by", options=["None"] + input_features, key="size_feature")
        
        # Display selected visualization
        if graph_type == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Non-Diabetic', 'Diabetic'],
                y=['Non-Diabetic', 'Diabetic'],
                text_auto=True,
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        elif graph_type == "ROC Curve":
            st.subheader("ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                showlegend=True
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        elif graph_type == "Feature Importance":
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=True)
            fig_importance = px.bar(
                importance_df, x='Importance', y='Feature', orientation='h',
                title=f"Feature Importance ({algorithm})",
                color='Importance', color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        elif graph_type == "Scatter Plot":
            st.subheader(f"{x_feature} vs {y_feature}")
            fig_scatter = px.scatter(
                data,
                x=x_feature,
                y=y_feature,
                color=color_feature,
                size=size_feature if size_feature != "None" else None,
                title=f"{x_feature} vs {y_feature} (Colored by {color_feature})",
                labels={x_feature: x_feature, y_feature: y_feature, color_feature: color_feature}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    main()
