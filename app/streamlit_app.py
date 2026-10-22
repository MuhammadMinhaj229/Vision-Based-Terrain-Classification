import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.terrain_classifier import TerrainClassifier, TERRAIN_PROFILES

st.set_page_config(
    page_title="UGV Terrain Classification & Rover Traversal Telemetry",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark Mission-Control Styling - Zero Emojis
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .kpi-card {
        background: linear-gradient(135deg, #161f30 0%, #101624 100%);
        border: 1px solid #23314d;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 15px;
    }
    .kpi-title {
        color: #8b9bb4;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 6px;
    }
    .action-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-cruise { background-color: #166534; color: #86efac; border: 1px solid #22c55e; }
    .badge-throttle { background-color: #854d0e; color: #fef08a; border: 1px solid #eab308; }
    .badge-lock { background-color: #9a3412; color: #fed7aa; border: 1px solid #f97316; }
    .badge-halt { background-color: #991b1b; color: #fecaca; border: 1px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return TerrainClassifier()

classifier = load_engine()

# Sidebar: Mission Configuration
with st.sidebar:
    st.markdown("## Ground Control Telemetry")
    st.caption("Autonomous Rover Traversability & Soil Mechanics Mapping")
    st.markdown("---")

    input_mode = st.radio(
        "Optical Patch Ingestion",
        options=["Preloaded Physical Benchmarks (SIH_NN)", "Upload Field Camera Image"],
        index=0
    )

    selected_image = None
    dataset_base = os.path.join(os.path.dirname(__file__), "..", "SIH_NN")

    if input_mode == "Preloaded Physical Benchmarks (SIH_NN)":
        terrain_category = st.selectbox(
            "Select Benchmark Terrain",
            options=["Desert (Sand Dunes)", "Forest (Vegetation & Canopy)", "Mountain (Rocky Scree)", "Plains (Grassland)"]
        )
        folder_map = {
            "Desert (Sand Dunes)": "Desert",
            "Forest (Vegetation & Canopy)": "Forest",
            "Mountain (Rocky Scree)": "Mountain",
            "Plains (Grassland)": "Plains"
        }
        folder_name = folder_map[terrain_category]
        target_dir = os.path.join(dataset_base, folder_name)
        if os.path.exists(target_dir):
            file_candidates = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if file_candidates:
                chosen_file = file_candidates[0]
                img_path = os.path.join(target_dir, chosen_file)
                selected_image = Image.open(img_path)
                st.success(f"Loaded ground patch: {chosen_file}")
    else:
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)

    st.markdown("---")
    st.caption("Engine: Spectral Gradient & Texture Telemetry | Zero Emojis Enforced")

# Main Header
st.title("Autonomous UGV Terrain Classification & Traversability Telemetry")
st.markdown("Quantify ground surface properties, compute soil traction coefficients, and inject traversability penalties into ROS 2 Nav2 global costmaps.")

if selected_image is not None:
    # Run Inference
    with st.spinner("Analyzing spectral signatures and computing soil mechanics..."):
        analysis = classifier.classify_image(selected_image)

    predicted_class = analysis["predicted_class"]
    confidence_pct = analysis["confidence"] * 100
    soil = analysis["soil_mechanics"]
    directive = analysis["navigation_directive"]

    # Executive KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Classified Surface</div>
            <div class="kpi-value">{predicted_class}</div>
            <div class="kpi-subtext">Confidence: {confidence_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Traction Friction Coeff (mu)</div>
            <div class="kpi-value" style="color: #38bdf8;">{soil['friction_coefficient_mu']:.2f}</div>
            <div class="kpi-subtext">Nominal rolling contact</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Wheel Slip & Sinkage Hazard</div>
            <div class="kpi-value" style="color: {'#ef4444' if soil['sinkage_risk_percent'] > 50 else '#22c55e'};">{soil['sinkage_risk_percent']}%</div>
            <div class="kpi-subtext">Bekker-Wong soil model</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Max Safe Traverse Speed</div>
            <div class="kpi-value">{soil['max_safe_speed_kmh']:.1f} km/h</div>
            <div class="kpi-subtext">Nav2 Costmap Cost: {soil['nav2_costmap_cost']}/100</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    tab_patch, tab_probabilities, tab_profiles, tab_ros2 = st.tabs([
        "Visual Patch & Ground Directive",
        "Probabilistic Softmax Distribution",
        "Soil Mechanics Reference Matrix",
        "ROS 2 Nav2 Costmap Architecture"
    ])

    with tab_patch:
        col_img, col_dir = st.columns([1, 1.3])
        with col_img:
            st.image(selected_image, caption="Optical Sensor Input Patch", use_container_width=True)

        with col_dir:
            st.markdown("### Locomotion Directive")
            action_code = directive["action"]
            badge_class = "badge-cruise" if action_code == "NORMAL_CRUISE" else "badge-throttle" if "THROTTLE" in action_code else "badge-lock" if "DIFF" in action_code else "badge-halt"
            st.markdown(f'<span class="action-badge {badge_class}">{action_code}</span>', unsafe_allow_html=True)
            st.markdown(f"<br><p style='font-size:1.05rem; color:#cbd5e1;'>{directive['advisory']}</p>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Surface Texture Diagnostics")
            tex = analysis["texture_features"]
            t1, t2, t3 = st.columns(3)
            t1.metric("Gradient Roughness", f"{tex['roughness_metric']}")
            t2.metric("Mean Luminance", f"{tex['mean_brightness']}")
            t3.metric("Spectral Saturation", f"{tex['saturation']}")

    with tab_probabilities:
        st.subheader("Classification Confidence Across Terrain Profiles")
        prob_df = pd.DataFrame(list(analysis["class_probabilities"].items()), columns=["Terrain Class", "Probability"])
        prob_df = prob_df.sort_values(by="Probability", ascending=True)

        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Terrain Class",
            orientation="h",
            color="Probability",
            color_continuous_scale="Blues",
            title="Model Softmax Probability Distribution"
        )
        fig_prob.update_layout(
            plot_bgcolor="#161f30",
            paper_bgcolor="#0b0f19",
            font=dict(color="#cbd5e1"),
            xaxis=dict(gridcolor="#23314d", range=[0, 1.0]),
            yaxis=dict(gridcolor="#23314d")
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with tab_profiles:
        st.subheader("Physical Soil Mechanics & Locomotion Specifications")
        matrix_rows = []
        for name, p in TERRAIN_PROFILES.items():
            matrix_rows.append({
                "Terrain Class": name,
                "Description": p["description"],
                "Friction Coeff (mu)": p["friction_coefficient"],
                "Sinkage Risk (%)": p["sinkage_risk_percent"],
                "Max Velocity (km/h)": p["max_safe_speed_kmh"],
                "Nav2 Costmap Penalty": p["nav2_costmap_cost"],
                "Locomotion Action": p["nav_action"]
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    with tab_ros2:
        st.subheader("ROS 2 Nav2 Navigation Stack Integration")
        st.markdown("""
        In real-time robotic systems, the output of this vision model is published to standard ROS 2 message topics:
        * **`/terrain/classification`** (`std_msgs/String`): Publishes classified surface category at 10Hz.
        * **`/terrain/traversability_cost`** (`nav2_msgs/CostmapFilterInfo`): Injected dynamically as a Costmap Layer into Nav2 Costmap 2D.
        * **`/cmd_vel/throttle_limit`** (`geometry_msgs/Twist`): Throttles maximum linear velocity ($v_{\max}$) to prevent wheel entrapment in high-sinkage sand or mud.
        """)
        st.code(f"""
# Sample ROS 2 Nav2 Costmap Layer Configuration
terrain_costmap_filter:
  plugin: "nav2_costmap_2d::CostmapFilter"
  filter_info_topic: "/terrain/traversability_cost"
  transform_tolerance: 0.1
  current_terrain: "{predicted_class}"
  injected_cost: {soil['nav2_costmap_cost']}
  friction_coefficient_mu: {soil['friction_coefficient_mu']}
  enforced_speed_limit_mps: {soil['max_safe_speed_kmh'] / 3.6:.2f}
        """, language="yaml")
else:
    st.info("Select a benchmark patch or upload a terrain image from the sidebar to start inference.")
