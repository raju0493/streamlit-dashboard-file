import streamlit as st

def show_about_page():
    # --- Custom CSS for stunning visual design ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #eef9f2 100%);
        }
        
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #eef9f2 100%);
        }
        
        /* Gorgeous header with glass effect */
        .glass-header {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            padding: 2.5em 3em;
            margin-bottom: 3em;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            gap: 2.5rem;
            transform: translateY(0);
            transition: all 0.5s ease;
            animation: fadeInUp 0.8s ease-out forwards;
        }
        
        .glass-header:hover {
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
            transform: translateY(-5px);
        }
        
        /* Beautiful card design with animation */
        .hero-card {
            background: linear-gradient(135deg, #e7f6fd 0%, #c8f5e2 100%);
            border-radius: 24px;
            padding: 3em; 
            margin-bottom: 3em;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06);
            border-left: 8px solid #10b981;
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s 0.2s ease-out forwards;
            opacity: 0;
            transform: translateY(30px);
        }
        
        .hero-card::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.2) 100%);
            transform: rotate(45deg);
            z-index: 0;
            transition: all 1.5s ease;
        }
        
        .hero-card:hover::before {
            top: 150%;
            right: 150%;
        }
        
        /* Gorgeous tag styling */
        .tag {
            display: inline-block;
            padding: 0.4em 1em;
            border-radius: 50px;
            margin-right: 0.7em;
            margin-bottom: 0.7em;
            font-weight: 600;
            font-size: 1em;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        
        .tag:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.12);
        }
        
        /* Beautiful feature cards */
        .feature-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 1.8em 1.6em;
            margin-bottom: 1.5em; 
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            border-top: 6px solid #10b981;
            min-height: 180px;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.6s ease forwards;
        }
        
        .feature-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 35px rgba(0, 0, 0, 0.1);
        }
        
        .feature-icon {
            font-size: 2.8em;
            margin-bottom: 0.3em;
            background: linear-gradient(135deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
        }
        
        /* Beautiful section styling */
        .section-title {
            color: #0f766e;
            font-weight: 700;
            letter-spacing: 0.02em;
            font-size: 2em;
            margin-bottom: 1em;
            position: relative;
            display: inline-block;
            padding-bottom: 0.5em;
        }
        
        .section-title::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #10b981, transparent);
            border-radius: 2px;
        }
        
        /* Gradient text styling */
        .gradient-header {
            font-size: 3.2em;
            font-weight: 800;
            line-height: 1.2;
            background: linear-gradient(135deg, #0f766e 0%, #10b981 50%, #3b82f6 100%);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.2em;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Beautiful dashboard cards */
        .dashboard-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.06);
            border-left: 5px solid;
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
            opacity: 0;
            transform: translateY(30px);
            animation: fadeInUp 0.5s ease forwards;
        }
        
        .dashboard-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .dashboard-icon {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #0f766e, #3b82f6);
            -webkit-background-clip: text;
            color: transparent;
            display: block;
        }
        
        .dashboard-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            color: #0f766e;
        }
        
        /* Steps styling */
        .step-container {
            position: relative;
            padding-left: 3.5em;
            margin-bottom: 2em;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.6s ease forwards;
        }
        
        .step-number {
            position: absolute;
            left: 0;
            top: 0;
            width: 2.5em;
            height: 2.5em;
            background: linear-gradient(135deg, #0f766e, #10b981);
            border-radius: 50%;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2em;
            box-shadow: 0 8px 16px rgba(16, 185, 129, 0.2);
        }
        
        .step-content {
            background: white;
            padding: 1.5em 2em;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
            position: relative;
            z-index: 1;
            margin-left: 1em;
            border-left: 4px solid #10b981;
        }
        
        .step-title {
            font-weight: 700;
            color: #0f766e;
            font-size: 1.3em;
            margin-bottom: 0.5em;
        }
        
        /* Terminology table */
        .term-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-bottom: 2em;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        }
        
        .term-table th {
            background: linear-gradient(135deg, #0f766e 0%, #10b981 100%);
            color: white;
            padding: 16px;
            text-align: left;
            font-weight: 600;
        }
        
        .term-table td {
            padding: 16px;
            border-bottom: 1px solid #e5e7eb;
            background-color: white;
        }
        
        .term-table tr:last-child td {
            border-bottom: none;
        }
        
        .term-table tr:hover td {
            background-color: #f8fafc;
        }
        
        /* Getting started items */
        .getting-started-item {
            display: flex;
            align-items: center;
            padding: 1.2em 1.5em;
            margin-bottom: 1em;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
            border-left: 4px solid #10b981;
            transition: all 0.3s ease;
            transform: translateX(0);
        }
        
        .getting-started-item:hover {
            transform: translateX(10px);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.08);
        }
        
        .getting-started-icon {
            background: linear-gradient(135deg, #0f766e, #10b981);
            color: white;
            width: 2.2em;
            height: 2.2em;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1.2em;
            font-size: 1.1em;
            box-shadow: 0 6px 12px rgba(16, 185, 129, 0.15);
        }
        
        /* Contact card */
        .contact-card {
            background: linear-gradient(135deg, #0f766e 0%, #10b981 100%);
            color: white;
            border-radius: 20px;
            padding: 2.5em;
            margin-top: 3em;
            box-shadow: 0 20px 40px rgba(15, 118, 110, 0.2);
            position: relative;
            overflow: hidden;
            text-align: center;
            animation: fadeInUp 0.8s 0.4s ease-out forwards;
            opacity: 0;
        }
        
        .contact-card::before {
            content: "";
            position: absolute;
            width: 300px;
            height: 300px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            top: -150px;
            right: -150px;
        }
        
        .contact-card::after {
            content: "";
            position: absolute;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            bottom: -100px;
            left: -100px;
        }
        
        .contact-button {
            display: inline-block;
            background: white;
            color: #0f766e;
            padding: 0.9em 2em;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1em;
            text-decoration: none;
            margin-top: 1.5em;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .contact-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
        }
        
        /* Model highlights card */
        .highlight-card {
            background: linear-gradient(135deg, #e7f6fd 0%, #c8f5e2 100%);
            border-radius: 20px;
            padding: 2em;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.06);
            margin-top: 2em;
            border: 1px solid rgba(255, 255, 255, 0.4);
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s 0.3s ease-out forwards;
            opacity: 0;
        }
        
        .highlight-card::before {
            content: "";
            position: absolute;
            top: -100px;
            left: -100px;
            width: 200px;
            height: 200px;
            background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 100%);
            border-radius: 50%;
            opacity: 0.4;
        }
        
        .highlight-title {
            color: #0f766e;
            font-weight: 700;
            font-size: 1.4em;
            margin-bottom: 1em;
            position: relative;
        }
        
        .highlight-list {
            padding-left: 1.5em;
            margin-top: 1em;
            position: relative;
            z-index: 1;
        }
        
        .highlight-list li {
            margin-bottom: 0.7em;
            position: relative;
            padding-left: 1.5em;
        }
        
        .highlight-list li::before {
            content: "✓";
            position: absolute;
            left: 0;
            top: 0;
            color: #10b981;
            font-weight: bold;
        }
        
        /* Animated icons */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Animation delays for staggered entrance */
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.2s; }
        .delay-3 { animation-delay: 0.3s; }
        .delay-4 { animation-delay: 0.4s; }
        .delay-5 { animation-delay: 0.5s; }
        .delay-6 { animation-delay: 0.6s; }
        </style>
    """, unsafe_allow_html=True)

    # --- Stunning Header with Logo ---
    st.markdown(
        """
        <div class="glass-header">
            <img src="https://img.icons8.com/fluency/96/000000/energy-meter.png" width="85">
            <div>
                <h1 class="gradient-header">Building Energy Analysis Dashboard</h1>
                <p style="font-size: 1.5em; color: #0f766e; font-weight: 500; margin-top: 0.5em;">
                    Transforming Buildings. Empowering Decisions. Securing Our Future.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Beautiful Hero Card ---
    st.markdown(
        """
        <div class="hero-card">
            <h2 style="color: #0f766e; font-size: 2.2em; margin-top: 0; margin-bottom: 1em; font-weight: 700;">
                Your Intelligent Energy Co-Pilot
            </h2>
            <div>
            <p style="font-size: 1.25em; line-height: 1.7; margin-bottom: 1.8em; color: #1e293b; max-width: 90%;">
                Welcome to the <b>Building Energy Analysis Dashboard</b>, your comprehensive platform for optimizing 
                building energy performance. Harness the power of advanced AI to transform how you design, 
                retrofit, and manage buildings for superior energy efficiency and sustainability.
            </p>
            </div>
            <div style="margin-bottom: 2em;">
                <span class="tag" style="background: linear-gradient(135deg, #0f766e 0%, #10b981 100%); color: white;">Predict</span>
                <span class="tag" style="background: linear-gradient(135deg, #047857 0%, #34d399 100%); color: white;">Analyze</span>
                <span class="tag" style="background: linear-gradient(135deg, #0369a1 0%, #38bdf8 100%); color: white;">Visualize</span>
                <span class="tag" style="background: linear-gradient(135deg, #7c3aed 0%, #c4b5fd 100%); color: white;">Evaluate</span>
                <span class="tag" style="background: linear-gradient(135deg, #be123c 0%, #fb7185 100%); color: white;">Report</span>
            </div>
            <div>       
            <div style="background: rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 1.8em; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.4);">
                <ul style="font-size: 1.2em; line-height: 2; padding-left: 0.5em; list-style-type: none; margin-bottom: 0;">
                    <li style="margin-bottom: 0.8em; display: flex; align-items: center; gap: 0.8em;">
                        <span style="background: linear-gradient(135deg, #0f766e 0%, #10b981 100%); color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold;">1</span>
                        <span><strong style="color: #0f766e;">Predict</strong> energy use with pinpoint accuracy for any building scenario</span>
                    </li>
                    <li style="margin-bottom: 0.8em; display: flex; align-items: center; gap: 0.8em;">
                        <span style="background: linear-gradient(135deg, #047857 0%, #34d399 100%); color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold;">2</span>
                        <span><strong style="color: #047857;">Analyze</strong> the impact of upgrades, retrofits, and design modifications</span>
                    </li>
                    <li style="margin-bottom: 0.8em; display: flex; align-items: center; gap: 0.8em;">
                        <span style="background: linear-gradient(135deg, #0369a1 0%, #38bdf8 100%); color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold;">3</span>
                        <span><strong style="color: #0369a1;">Visualize</strong> the key drivers behind your building's energy profile</span>
                    </li>
                    <li style="margin-bottom: 0.8em; display: flex; align-items: center; gap: 0.8em;">
                        <span style="background: linear-gradient(135deg, #7c3aed 0%, #c4b5fd 100%); color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold;">4</span>
                        <span><strong style="color: #7c3aed;">Evaluate</strong> ROI, payback periods, and CO₂ impact of efficiency measures</span>
                    </li>
                    <li style="display: flex; align-items: center; gap: 0.8em;">
                        <span style="background: linear-gradient(135deg, #be123c 0%, #fb7185 100%); color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold;">5</span>
                        <span><strong style="color: #be123c;">Generate</strong> comprehensive reports for stakeholders and clients</span>
                    </li>
                </ul>
            </div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Feature Cards ---
    st.markdown('<h2 class="section-title">Powerful Features</h2>', unsafe_allow_html=True)
    
    # Create two rows of 3 cards each for better layout
    for row in range(2):
        cols = st.columns(3)
        for i in range(3):
            idx = row * 3 + i
            delay = idx + 1
            
            features = [
                {
                    "icon": "⚡", 
                    "title": "Energy Usage Prediction",
                    "desc": "Advanced machine learning models deliver accurate energy consumption forecasts based on building characteristics.",
                    "color": "#0f766e"
                },
                {
                    "icon": "🔄", 
                    "title": "Interactive What-If Analysis",
                    "desc": "Experiment with building parameters in real-time to see immediate energy performance impacts.",
                    "color": "#047857"
                },
                {
                    "icon": "🧠", 
                    "title": "Explainable AI Visualizations",
                    "desc": "Understand exactly what drives your building's energy use with transparent, interpretable AI.",
                    "color": "#0369a1"
                },
                {
                    "icon": "💡", 
                    "title": "Retrofit Recommendations",
                    "desc": "Receive AI-powered, prioritized suggestions for the most impactful building improvements.",
                    "color": "#7c3aed"
                },
                {
                    "icon": "💰", 
                    "title": "Cost-Benefit Analysis",
                    "desc": "Comprehensive financial evaluation of energy efficiency investments with ROI calculations.",
                    "color": "#be123c"
                },
                {
                    "icon": "📄", 
                    "title": "Automated Report Generation",
                    "desc": "Create professional-grade reports with a single click for stakeholders and compliance.",
                    "color": "#0e7490"
                }
            ]
            
            if idx < len(features):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="feature-card delay-{delay}" style="animation-delay: {0.1 * delay}s; border-top: 6px solid {features[idx]['color']};">
                            <span class="feature-icon" style="background: linear-gradient(135deg, {features[idx]['color']}, {features[idx]['color']}30);">{features[idx]['icon']}</span>
                            <h3 style="font-weight: 700; font-size: 1.25em; color: {features[idx]['color']}; margin-bottom: 0.6em;">{features[idx]['title']}</h3>
                            <p style="color: #334155; line-height: 1.6;">{features[idx]['desc']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # --- How It Works with Beautiful Step Cards ---
    st.markdown('<h2 class="section-title">How It Works</h2>', unsafe_allow_html=True)
    
    steps = [
        {
            "title": "Input Your Building Details",
            "desc": "Enter specifications like insulation values, HVAC efficiency, and occupancy patterns.",
            "icon": "1"
        },
        {
            "title": "AI Analysis",
            "desc": "Our sophisticated LightGBM AI model processes your data through advanced algorithms.",
            "icon": "2"
        },
        {
            "title": "Review Results",
            "desc": "Instantly see energy consumption, cost implications, and carbon footprint predictions.",
            "icon": "3"
        },
        {
            "title": "Explore Improvements",
            "desc": "Receive tailored recommendations with detailed ROI and environmental impact analyses.",
            "icon": "4"
        }
    ]
    
    for i, step in enumerate(steps):
        st.markdown(
            f"""
            <div class="step-container" style="animation-delay: {0.15 * (i+1)}s;">
                <div class="step-number">{step['icon']}</div>
                <div class="step-content">
                    <div class="step-title">{step['title']}</div>
                    <div style="color: #475569;">{step['desc']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Model highlights card with beautiful design
    st.markdown(
        """
        <div class="highlight-card">
            <h3 class="highlight-title">AI Model Highlights</h3>
            <p style="color: #334155; line-height: 1.7; position: relative; z-index: 1;">
                Our state-of-the-art machine learning model delivers exceptional performance through:
            </p>
            <ul class="highlight-list">
                <li><strong>Comprehensive Training</strong> on thousands of real and simulated buildings</li>
                <li><strong>Advanced Algorithms</strong> that capture complex, non-linear relationships</li>
                <li><strong>Real-Time Processing</strong> for immediate actionable insights</li>
                <li><strong>High Accuracy</strong> validated against real-world energy consumption data</li>
                <li><strong>Interpretable Results</strong> with clear explanations of prediction factors</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # --- Dashboard Sections with Beautiful Cards ---
    st.markdown('<h2 class="section-title">Interactive Dashboard Sections</h2>', unsafe_allow_html=True)

    # Define sections with icons, titles, descriptions and colors
    sections = [
        {
            "icon": "🏠",
            "title": "Dashboard Overview",
            "desc": "Comprehensive summary of key energy metrics and performance indicators.",
            "color": "#0f766e"
        },
        {
            "icon": "🏢",
            "title": "Building Details",
            "desc": "In-depth building specifications with benchmarking against industry standards.",
            "color": "#047857"
        },
        {
            "icon": "🧩",
            "title": "Model Explanation",
            "desc": "Interactive visualizations showing which factors influence energy consumption most.",
            "color": "#0369a1"
        },
        {
            "icon": "🔬",
            "title": "What-If Analysis",
            "desc": "Real-time experimentation with building parameters to optimize performance.",
            "color": "#7c3aed"
        },
        {
            "icon": "🔧",
            "title": "Retrofit Analysis",
            "desc": "Detailed evaluation of potential upgrades with financial and environmental metrics.",
            "color": "#be123c"
        },
        {
            "icon": "🌱",
            "title": "Recommendations",
            "desc": "Prioritized action steps to improve building energy efficiency and sustainability.",
            "color": "#0e7490"
        }
    ]

    # Create 3 columns for cards with animation delays
    cols = st.columns(3)
    for idx, section in enumerate(sections):
        col = cols[idx % 3]
        delay = idx + 1
        
        col.markdown(f"""
            <div class="dashboard-card delay-{delay}" style="animation-delay: {0.15 * delay}s; border-left-color: {section['color']};">
                <span class="dashboard-icon" style="background: linear-gradient(135deg, {section['color']}, {section['color']}50);">{section['icon']}</span>
                <h3 class="dashboard-title" style="color: {section['color']};">{section['title']}</h3>
                <p style="color: #475569; line-height: 1.7;">{section['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- Key Terminology Section with Tabs and Beautiful Tables ---
    st.markdown('<h2 class="section-title">Key Terminology</h2>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Energy & Building Terms", "Financial & Retrofit Terms"])
    
    with tabs[0]:
        st.markdown("""
        <table class="term-table">
            <thead>
                <tr>
                    <th>Term</th>
                    <th>Definition</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Energy Use Intensity (EUI)</strong></td>
                    <td>Annual energy consumption per unit area (kWh/m²/year). Lower values indicate better energy efficiency.</td>
                </tr>
                <tr>
                    <td><strong>U-Value</strong></td>
                    <td>Thermal transmittance measured in W/m²K. Lower values indicate better insulation properties.</td>
                </tr>
                <tr>
                    <td><strong>HVAC Efficiency</strong></td>
                    <td>Coefficient of Performance (COP) for heating, ventilation, and air conditioning systems. Higher values indicate greater efficiency.</td>
                </tr>
                <tr>
                    <td><strong>Air Change Rate</strong></td>
                    <td>Frequency of air replacement in a space, measured in air changes per hour (ACH). Lower values typically indicate better airtightness.</td>
                </tr>
                <tr>
                    <td><strong>Window to Wall Ratio</strong></td>
                    <td>Proportion of wall area covered by windows. Higher values typically increase solar gain but may affect thermal performance.</td>
                </tr>
                <tr>
                    <td><strong>CO₂ Emissions</strong></td>
                    <td>Carbon dioxide released from energy consumption, calculated using standardized emission factors.</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
    with tabs[1]:
        st.markdown("""
        <table class="term-table">
            <thead>
                <tr>
                    <th>Term</th>
                    <th>Definition</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Simple Payback Period</strong></td>
                    <td>Time required for savings to equal the initial investment, measured in years.</td>
                </tr>
                <tr>
                    <td><strong>Return on Investment (ROI)</strong></td>
                    <td>Financial gain relative to investment cost, expressed as a percentage.</td>
                </tr>
                <tr>
                    <td><strong>Net Present Value (NPV)</strong></td>
                    <td>Current value of all future cash flows generated by an investment, accounting for the time value of money.</td>
                </tr>
                <tr>
                    <td><strong>Implementation Cost</strong></td>
                    <td>Total upfront expenditure required for an energy efficiency upgrade or retrofit.</td>
                </tr>
                <tr>
                    <td><strong>Annual Cost Savings</strong></td>
                    <td>Yearly reduction in energy expenses resulting from implemented efficiency measures.</td>
                </tr>
                <tr>
                    <td><strong>Lifetime CO₂ Reduction</strong></td>
                    <td>Total carbon emissions avoided over the complete lifespan of an efficiency measure.</td>
                </tr>
                <tr>
                    <td><strong>Carbon Equivalence</strong></td>
                    <td>Translation of carbon reduction into relatable terms (e.g., equivalent trees planted or cars removed).</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    # --- Getting Started with Beautiful List Items ---
    st.markdown('<h2 class="section-title">Getting Started</h2>', unsafe_allow_html=True)
    
    getting_started_steps = [
        ("Dashboard Overview", "Get a comprehensive view of your building's energy performance metrics"),
        ("Building Details", "Review and update your building's specifications and characteristics"),
        ("Model Explanation", "Understand the key factors driving your building's energy consumption"),
        ("What-If Analysis", "Experiment with different parameters to optimize energy efficiency"),
        ("Retrofit Analysis", "Evaluate potential upgrades and their financial and environmental impacts"),
        ("Recommendations", "Review personalized suggestions for improving building performance"),
        ("Generate Reports", "Create professional documentation for stakeholders and compliance")
    ]
    
    icons = ["📊", "🏢", "📈", "🔍", "🔧", "✅", "📑"]
    
    for i, (title, desc) in enumerate(getting_started_steps):
        st.markdown(
            f"""
            <div class="getting-started-item delay-{i+1}" style="animation-delay: {0.1 * (i+1)}s;">
                <div class="getting-started-icon">{icons[i]}</div>
                <div>
                    <strong style="color: #0f766e; font-size: 1.15em;">{title}</strong>
                    <p style="margin: 0.3em 0 0 0; color: #475569;">{desc}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Pro Tips Section with Beautiful Design
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 20px; padding: 2em; margin-top: 2em; box-shadow: 0 15px 30px rgba(0, 0, 0, 0.05); border-left: 5px solid #10b981; animation: fadeInUp 0.8s 0.5s ease-out forwards; opacity: 0;">
            <div style="display: flex; align-items: center; margin-bottom: 1em;">
                <span style="font-size: 2em; margin-right: 0.5em;">💡</span>
                <h3 style="color: #0f766e; margin: 0; font-size: 1.5em;">Pro Tips for Maximum Impact</h3>
            </div>
            <ul style="margin: 0; padding-left: 1.5em; color: #334155; font-size: 1.1em; line-height: 1.8;">
                <li><strong>Enter accurate details</strong> for the most precise energy predictions</li>
                <li><strong>Save different scenarios</strong> to compare multiple design or retrofit options</li>
                <li><strong>Export comprehensive reports</strong> in PDF format for professional presentations</li>
                <li><strong>Use sensitivity analysis</strong> to identify the most influential building parameters</li>
                <li><strong>Combine recommendations</strong> into packages for maximum energy and cost savings</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Beautiful Contact Section ---
    st.markdown(
        """
        <div class="contact-card">
            <h3 style="font-size: 1.8em; margin-top: 0; font-weight: 700;">We Value Your Feedback</h3>
            <p style="font-size: 1.2em; max-width: 80%; margin: 1em auto; line-height: 1.7;">
                Have questions, suggestions, or need assistance with your building energy analysis?
                Our team of experts is ready to help you optimize your building's performance.
            </p>
            <a href="mailto:support@example.com" class="contact-button">
                Contact Our Support Team
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )

def get_feature_description(feature):
    descriptions = {
        "Energy Usage Prediction": "Forecast energy consumption based on building characteristics and operational patterns.",
        "Interactive What-If Analysis": "Experiment with building parameters to see real-time impact on energy performance.",
        "Explainable AI Visualizations": "Understand exactly what drives your building's energy consumption through transparent AI.",
        "Retrofit Recommendations": "Receive AI-powered suggestions for the most effective building improvements.",
        "Cost-Benefit Analysis": "Evaluate the financial returns and environmental benefits of energy efficiency investments.",
        "Automated Report Generation": "Create professional documentation for stakeholders with a single click."
    }
    return descriptions.get(feature, "")