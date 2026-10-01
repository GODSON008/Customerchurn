let chartRetentionDonut = null;
let chartRegionChurn = null;
let chartTenureTrend = null;
let chartTenureRevenue = null;
let chartContractDriver = null;
let chartTicketsDriver = null;

document.addEventListener("DOMContentLoaded", () => {
    loadKPIs();
    loadTrends();
    loadDrivers();
    loadAnalystReport();
    updateSimulation();
});

function switchTab(tabId) {
    document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
    
    const targetTab = document.getElementById(`tab-${tabId}`);
    if (targetTab) {
        targetTab.classList.add("active");
    }
    
    // Find button
    const buttons = document.querySelectorAll(".nav-item");
    buttons.forEach(btn => {
        if (btn.getAttribute("onclick") && btn.getAttribute("onclick").includes(tabId)) {
            btn.classList.add("active");
        }
    });
}

function formatCurrency(val) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);
}

// 1. Load Top Level KPIs
async function loadKPIs() {
    try {
        const res = await fetch('/api/kpis');
        const data = await res.json();
        
        document.getElementById('kpi-total-revenue').innerText = formatCurrency(data.total_revenue);
        document.getElementById('kpi-churn-rate').innerText = `${data.churn_rate_pct}%`;
        document.getElementById('kpi-churn-count').innerText = data.churned_customers.toLocaleString();
        document.getElementById('kpi-risk-revenue').innerText = formatCurrency(data.annual_revenue_at_risk);
        document.getElementById('kpi-retention-savings').innerText = formatCurrency(data.projected_retention_savings);
        
        renderDonutChart(data.active_customers, data.churned_customers);
    } catch (err) {
        console.error("Error loading KPIs:", err);
    }
}

// 2. Render Donut Chart
function renderDonutChart(active, churned) {
    const ctx = document.getElementById('chartRetentionDonut').getContext('2d');
    if (chartRetentionDonut) chartRetentionDonut.destroy();
    
    chartRetentionDonut = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Retained Subscribers', 'Churned Accounts'],
            datasets: [{
                data: [active, churned],
                backgroundColor: ['#3b82f6', '#ef4444'],
                borderWidth: 0,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 12 } }
                }
            },
            cutout: '70%'
        }
    });
}

// 3. Load Trends & Region data
async function loadTrends() {
    try {
        const res = await fetch('/api/analytics/trends');
        const data = await res.json();
        
        // Region Chart
        const regionLabels = data.region_summary.map(r => r.Region);
        const regionChurnRates = data.region_summary.map(r => r.churn_rate);
        
        const ctxRegion = document.getElementById('chartRegionChurn').getContext('2d');
        if (chartRegionChurn) chartRegionChurn.destroy();
        chartRegionChurn = new Chart(ctxRegion, {
            type: 'bar',
            data: {
                labels: regionLabels,
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: regionChurnRates,
                    backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                    x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        // Tenure Trend Chart
        const tenureLabels = data.tenure_cohorts.map(c => c.TenureGroup);
        const tenureRates = data.tenure_cohorts.map(c => c.churn_rate);
        const tenureRevenues = data.tenure_cohorts.map(c => c.revenue);
        
        const ctxTenure = document.getElementById('chartTenureTrend').getContext('2d');
        if (chartTenureTrend) chartTenureTrend.destroy();
        chartTenureTrend = new Chart(ctxTenure, {
            type: 'line',
            data: {
                labels: tenureLabels,
                datasets: [{
                    label: 'Cohort Churn Rate (%)',
                    data: tenureRates,
                    borderColor: '#f87171',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    fill: true,
                    tension: 0.35,
                    pointRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                    x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        // Tenure Revenue Chart
        const ctxTenRev = document.getElementById('chartTenureRevenue').getContext('2d');
        if (chartTenureRevenue) chartTenureRevenue.destroy();
        chartTenureRevenue = new Chart(ctxTenRev, {
            type: 'bar',
            data: {
                labels: tenureLabels,
                datasets: [{
                    label: 'Monthly Charges ($)',
                    data: tenureRevenues,
                    backgroundColor: '#3b82f6',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                    x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
    } catch (err) {
        console.error("Error loading trends:", err);
    }
}

// 4. Load Drivers & Features Table
async function loadDrivers() {
    try {
        const res = await fetch('/api/analytics/drivers');
        const data = await res.json();
        
        // Contract Driver Chart
        const contractLabels = data.contract.map(c => c.Contract);
        const contractChurn = data.contract.map(c => c.churn_rate);
        
        const ctxContract = document.getElementById('chartContractDriver').getContext('2d');
        if (chartContractDriver) chartContractDriver.destroy();
        chartContractDriver = new Chart(ctxContract, {
            type: 'bar',
            data: {
                labels: contractLabels,
                datasets: [{
                    label: 'Churn %',
                    data: contractChurn,
                    backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                    x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        // Tickets Driver Chart
        const ticketLabels = data.support_tickets.map(t => `${t.SupportTickets} Tickets`);
        const ticketChurn = data.support_tickets.map(t => t.churn_rate);
        
        const ctxTickets = document.getElementById('chartTicketsDriver').getContext('2d');
        if (chartTicketsDriver) chartTicketsDriver.destroy();
        chartTicketsDriver = new Chart(ctxTickets, {
            type: 'line',
            data: {
                labels: ticketLabels,
                datasets: [{
                    label: 'Churn % by Tickets',
                    data: ticketChurn,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                    x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        // Top Features Table
        const tbody = document.querySelector("#featuresTable tbody");
        tbody.innerHTML = "";
        
        data.model_feature_importance.forEach((feat, idx) => {
            const tr = document.createElement("tr");
            let interp = "Primary behavioral risk indicator";
            if (feat.Feature.includes("Contract")) interp = "Contract duration directly locks in retention lifetime.";
            if (feat.Feature.includes("Tenure")) interp = "Customer maturity strongly reduces voluntary churn.";
            if (feat.Feature.includes("SupportTickets")) interp = "Repeated friction triggers rapid contract cancellation.";
            if (feat.Feature.includes("TechSupport")) interp = "Active support coverage insulates high-speed internet accounts.";
            if (feat.Feature.includes("MonthlyCharges")) interp = "Price sensitivity creates vulnerability to competitor offers.";
            
            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong></td>
                <td><code>${feat.Feature}</code></td>
                <td><div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:${Math.round(feat.Importance * 200)}px; height:6px; background:#3b82f6; border-radius:3px;"></div>
                    <span>${feat.Importance.toFixed(4)}</span>
                </div></td>
                <td>${interp}</td>
            `;
            tbody.appendChild(tr);
        });
        
    } catch (err) {
        console.error("Error loading drivers:", err);
    }
}

// 5. Run ML Prediction
async function runPrediction() {
    const payload = {
        Region: document.getElementById('inputRegion').value,
        Gender: "Female",
        SeniorCitizen: 0,
        Partner: "No",
        Dependents: "No",
        Tenure: parseInt(document.getElementById('inputTenure').value),
        PhoneService: "Yes",
        MultipleLines: "No",
        InternetService: document.getElementById('inputInternet').value,
        OnlineSecurity: "No",
        OnlineBackup: "No",
        DeviceProtection: "No",
        TechSupport: document.getElementById('inputTechSupport').value,
        StreamingTV: "Yes",
        StreamingMovies: "Yes",
        Contract: document.getElementById('inputContract').value,
        PaperlessBilling: "Yes",
        PaymentMethod: document.getElementById('inputPayment').value,
        MonthlyCharges: parseFloat(document.getElementById('inputMonthlyCharges').value),
        TotalCharges: parseFloat(document.getElementById('inputMonthlyCharges').value) * parseInt(document.getElementById('inputTenure').value),
        SupportTickets: parseInt(document.getElementById('inputSupportTickets').value)
    };
    
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await res.json();
        
        document.getElementById('riskScoreVal').innerText = `${result.risk_score_pct}%`;
        const levelBadge = document.getElementById('riskLevelVal');
        levelBadge.innerText = `${result.risk_level} Risk`;
        levelBadge.className = `risk-label ${result.risk_level.toLowerCase()}`;
        
        document.getElementById('annualLossVal').innerText = formatCurrency(result.estimated_annual_loss_if_churned);
        
        const list = document.getElementById('prescriptiveList');
        list.innerHTML = "";
        result.prescriptive_actions.forEach(action => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>Action:</strong> ${action}`;
            list.appendChild(li);
        });
    } catch (err) {
        console.error("Prediction failed:", err);
    }
}

// 6. Update Retention Simulator
async function updateSimulation() {
    const discount = parseFloat(document.getElementById('simDiscount').value);
    const loyalty = parseFloat(document.getElementById('simLoyalty').value);
    const techBundle = document.getElementById('simTechBundle').checked;
    
    document.getElementById('sliderDiscountVal').innerText = `${discount}%`;
    document.getElementById('sliderLoyaltyVal').innerText = `${loyalty}%`;
    
    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contract_upgrade_discount: discount,
                loyalty_retention_discount: loyalty,
                free_tech_support_bundle: techBundle
            })
        });
        const sim = await res.json();
        
        document.getElementById('simNewChurn').innerText = `${sim.simulated_churn_rate_pct}%`;
        document.getElementById('simReductionPct').innerText = `↓ ${sim.churn_reduction_pct}% Overall Churn Reduction`;
        document.getElementById('simRevProtected').innerText = formatCurrency(sim.annual_revenue_protected);
        document.getElementById('simRoiPct').innerText = `+${sim.net_retention_roi_pct}% Net ROI`;
    } catch (err) {
        console.error("Simulation error:", err);
    }
}

// 7. Load AI Executive Analyst Report
async function loadAnalystReport() {
    try {
        const res = await fetch('/api/executive-insights');
        const data = await res.json();
        
        // Findings
        const findingsList = document.getElementById('findingsList');
        findingsList.innerHTML = "";
        data.key_findings.forEach(f => {
            const li = document.createElement('li');
            li.innerText = f;
            findingsList.appendChild(li);
        });
        
        // Risks
        const risksList = document.getElementById('risksList');
        risksList.innerHTML = "";
        data.risks.forEach(r => {
            const li = document.createElement('li');
            li.innerText = r;
            risksList.appendChild(li);
        });
        
        // Opportunities
        const oppsList = document.getElementById('opportunitiesList');
        oppsList.innerHTML = "";
        data.opportunities.forEach(o => {
            const li = document.createElement('li');
            li.innerText = o;
            oppsList.appendChild(li);
        });
        
        // Actions
        const actionsList = document.getElementById('actionsList');
        actionsList.innerHTML = "";
        data.recommended_actions.forEach(a => {
            const li = document.createElement('li');
            li.innerText = a;
            actionsList.appendChild(li);
        });
        
        // Model Comparison Table
        const modelTable = document.querySelector('#modelComparisonTable tbody');
        modelTable.innerHTML = "";
        for (const [alg, m] of Object.entries(data.model_evaluation)) {
            const tr = document.createElement('tr');
            const isBest = alg === data.best_model;
            tr.innerHTML = `
                <td><strong>${alg} ${isBest ? '<span class="badge badge-accent">Champion Model</span>' : ''}</strong></td>
                <td>${(m.Accuracy * 100).toFixed(2)}%</td>
                <td>${(m.Precision * 100).toFixed(2)}%</td>
                <td>${(m.Recall * 100).toFixed(2)}%</td>
                <td>${(m.F1_Score * 100).toFixed(2)}%</td>
                <td><strong>${m.ROC_AUC.toFixed(4)}</strong></td>
            `;
            modelTable.appendChild(tr);
        }
    } catch (err) {
        console.error("Error loading analyst report:", err);
    }
}
