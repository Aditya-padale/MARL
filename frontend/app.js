/**
 * app.js — Frontend logic for MARL City Simulator
 * Handles WebSocket connection, HTML5 Canvas rendering, Chart.js updates,
 * and UI interactions for the dashboard, policy engine, and inspector.
 * 
 * Built with Vanilla JS as requested (no React/Vue).
 */

// ═══════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════

const State = {
    ws: null,
    connected: false,
    
    // Simulation Data
    timestep: 0,
    speed: 'play',
    isPaused: false,
    
    // City & Agents
    cityData: null,
    agents: new Map(), // id -> agent data
    
    // Metrics & Charts
    metrics: null,
    charts: {
        inequality: null,
        macro: null,
        actions: null
    },
    
    // UI State
    selectedAgentId: null,
    inspectorOpen: false,
    
    // Canvas
    canvas: null,
    ctx: null,
    width: 1200,
    height: 900,
    lastFrameTime: 0,
    
    // Animation specific
    smoothAgents: new Map() // For lerping positions between WebSocket ticks
};

// Map backend emotion strings to FontAwesome icons
const EMOTION_ICONS = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😠",
    "celebration": "🎉",
    "shock": "😲",
    "medical": "🏥",
    "remittance": "💸",
    "social": "💬",
    "neutral": ""
};

// ═══════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initUI();
    initCanvas();
    initCharts();
    connectWebSocket();
    
    // Start render loop
    requestAnimationFrame(renderLoop);
});

function initUI() {
    // Tab Switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Remove active classes
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked
            const targetId = e.currentTarget.getAttribute('data-target');
            e.currentTarget.classList.add('active');
            document.getElementById(`panel-${targetId}`).classList.add('active');
        });
    });
    
    // Time Controls
    document.getElementById('btn-pause').addEventListener('click', () => sendCommand('speed', 'pause'));
    document.getElementById('btn-play').addEventListener('click', () => sendCommand('speed', 'play'));
    document.getElementById('btn-fast').addEventListener('click', () => sendCommand('speed', 'week'));
    document.getElementById('btn-very-fast').addEventListener('click', () => sendCommand('speed', 'month'));
    
    document.getElementById('btn-skip-month').addEventListener('click', () => sendCommand('skip', 30));
    document.getElementById('btn-skip-year').addEventListener('click', () => sendCommand('skip', 365));
    
    // Policy Engine
    document.getElementById('btn-apply-policy').addEventListener('click', () => {
        const text = document.getElementById('policy-input').value.trim();
        if (text) {
            sendCommand('policy', text);
            document.getElementById('btn-apply-policy').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            document.getElementById('btn-apply-policy').disabled = true;
        }
    });
    
    // Comparison Engine
    document.getElementById('btn-run-compare').addEventListener('click', () => {
        const text = document.getElementById('compare-policy-input').value.trim();
        const steps = parseInt(document.getElementById('compare-duration').value);
        if (text) {
            sendCommand('compare', text, steps);
            document.getElementById('compare-loading').classList.remove('hidden');
            document.getElementById('compare-results').classList.add('hidden');
            document.getElementById('btn-run-compare').disabled = true;
        }
    });
    
    // Export
    const exportBtn = document.getElementById('btn-export-csv');
    if (exportBtn) {
        exportBtn.addEventListener('click', async () => {
            try {
                const res = await fetch('http://localhost:8000/api/export/metrics', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({filename: "data/comparison_export.csv"})
                });
                if (res.ok) alert("Exported successfully to data/comparison_export.csv");
            } catch (e) {
                console.error("Export failed", e);
            }
        });
    }
}

function initCanvas() {
    State.canvas = document.getElementById('city-canvas');
    State.ctx = State.canvas.getContext('2d');
    
    // Setup click handler for agent inspection
    State.canvas.addEventListener('click', (e) => {
        const rect = State.canvas.getBoundingClientRect();
        
        // Calculate scale (canvas logical size vs display size)
        const scaleX = State.canvas.width / rect.width;
        const scaleY = State.canvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        handleCanvasClick(x, y);
    });
}

function initCharts() {
    // Common chart options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 }, // Disable animation for live updates
        plugins: {
            legend: {
                labels: { color: '#94a3b8', font: { family: 'Inter' } }
            }
        },
        scales: {
            x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
    };
    
    // Inequality Chart
    const ctxIneq = document.getElementById('chart-inequality').getContext('2d');
    State.charts.inequality = new Chart(ctxIneq, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Gini Coefficient',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, min: 0.2, max: 0.8 }
            }
        }
    });
    
    // Macro Chart
    const ctxMacro = document.getElementById('chart-macro').getContext('2d');
    State.charts.macro = new Chart(ctxMacro, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Poverty Rate (%)',
                    data: [],
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Median Wealth (k₹)',
                    data: [],
                    borderColor: '#10b981',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            ...commonOptions,
            scales: {
                x: commonOptions.scales.x,
                y: { type: 'linear', display: true, position: 'left', min: 0, max: 100 },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
            }
        }
    });
    
    // Agent Actions Chart
    const ctxActions = document.getElementById('chart-actions').getContext('2d');
    State.charts.actions = new Chart(ctxActions, {
        type: 'doughnut',
        data: {
            labels: ['Save', 'Spend', 'Invest', 'Trade/Work'],
            datasets: [{
                data: [25, 25, 25, 25],
                backgroundColor: [
                    '#3b82f6', // Save - Blue
                    '#ef4444', // Spend - Red
                    '#8b5cf6', // Invest - Purple
                    '#10b981'  // Trade - Green
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { position: 'right', labels: { color: '#94a3b8', boxWidth: 12 } }
            }
        }
    });
}

// ═══════════════════════════════════════════
// WEBSOCKET COMM
// ═══════════════════════════════════════════

function connectWebSocket() {
    // Determine WS URL based on current host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // const host = window.location.host; 
    // Hardcoded for local dev with separate ports
    const wsUrl = `ws://localhost:8000/ws`;
    
    updateConnectionStatus(false, "Connecting...");
    
    State.ws = new WebSocket(wsUrl);
    
    State.ws.onopen = () => {
        State.connected = true;
        updateConnectionStatus(true, "Connected");
        console.log("WebSocket connected");
    };
    
    State.ws.onclose = () => {
        State.connected = false;
        updateConnectionStatus(false, "Disconnected. Reconnecting...");
        console.log("WebSocket disconnected. Retrying in 3s...");
        setTimeout(connectWebSocket, 3000);
    };
    
    State.ws.onerror = (err) => {
        console.error("WebSocket error", err);
    };
    
    State.ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        } catch (e) {
            console.error("Error parsing WS message", e);
        }
    };
}

function sendCommand(action, value = null, steps = null) {
    if (!State.connected || !State.ws) return;
    
    const cmd = { action };
    
    if (action === 'speed' || action === 'skip') {
        cmd.value = value;
    } else if (action === 'inspect') {
        cmd.agent_id = value;
    } else if (action === 'policy') {
        cmd.text = value;
    } else if (action === 'compare') {
        cmd.text = value;
        cmd.steps = steps;
    }
    
    State.ws.send(JSON.stringify(cmd));
}

function updateConnectionStatus(connected, text) {
    const indicator = document.getElementById('connection-indicator');
    const textEl = document.getElementById('connection-text');
    
    if (connected) {
        indicator.classList.add('connected');
    } else {
        indicator.classList.remove('connected');
    }
    textEl.innerText = text;
}

// ═══════════════════════════════════════════
// MESSAGE HANDLING
// ═══════════════════════════════════════════

function handleServerMessage(msg) {
    switch (msg.type) {
        case 'full_state':
            handleFullState(msg);
            break;
        case 'diff':
            handleDiff(msg);
            break;
        case 'inspect_result':
            updateInspectorPanel(msg.data);
            break;
        case 'policy_result':
            handlePolicyResult(msg);
            break;
        case 'comparison_started':
            // Handled via button click UI directly, but could add global overlay
            break;
        case 'comparison_result':
            handleComparisonResult(msg.data);
            break;
        case 'comparison_error':
            alert("Comparison failed: " + msg.error);
            document.getElementById('compare-loading').classList.add('hidden');
            document.getElementById('btn-run-compare').disabled = false;
            break;
    }
}

function handleFullState(msg) {
    State.timestep = msg.timestep;
    State.speed = msg.speed;
    State.isPaused = msg.isPaused;
    State.cityData = msg.city;
    
    updateTopBar(msg);
    updateSpeedButtons(msg.speed);
    
    // Store all agents
    State.agents.clear();
    State.smoothAgents.clear();
    
    msg.agents.forEach(a => {
        State.agents.set(a.id, a);
        // Init smooth lerp state
        State.smoothAgents.set(a.id, { x: a.x, y: a.y, targetX: a.x, targetY: a.y });
    });
    
    // Update metrics & charts
    if (msg.metrics) updateDashboardMetrics(msg.metrics, msg.social_stats);
    if (msg.trends) updateCharts(msg.trends);

    // Render network graph
    if (msg.social_graph && document.getElementById('panel-network').classList.contains('active')) {
        renderNetworkGraph(msg.social_graph);
    } else if (msg.social_graph) {
        State.latestSocialGraph = msg.social_graph; // Cache for later
    }
    
    // Add events
    if (msg.events && msg.events.length > 0) {
        const feed = document.getElementById('feed-content');
        feed.innerHTML = ''; // Clear
        msg.events.forEach(e => addEventToFeed(e));
    }
    
    // Active policies
    if (msg.active_policies) {
        updateActivePoliciesList(msg.active_policies);
    }
    
    // If inspector is open, refresh data
    if (State.inspectorOpen && State.selectedAgentId !== null) {
        sendCommand('inspect', State.selectedAgentId);
    }
}

function handleDiff(msg) {
    State.timestep = msg.timestep;
    
    updateTopBar(msg);
    updateSpeedButtons(msg.speed);
    
    // Update agents that changed
    if (msg.agent_updates) {
        msg.agent_updates.forEach(update => {
            if (State.agents.has(update.id)) {
                const a = State.agents.get(update.id);
                // Update properties
                Object.assign(a, update);
                
                // Update smooth target
                const s = State.smoothAgents.get(update.id);
                if (s) {
                    s.targetX = update.x;
                    s.targetY = update.y;
                }
            }
        });
    }
    
    // Update metrics
    if (msg.metrics) {
        updateDashboardMetrics(msg.metrics);
        
        // Append to charts if we're not running too fast
        if (msg.timestep % 5 === 0) {
            appendChartData(msg);
        }
    }
    
    // Add events
    if (msg.events && msg.events.length > 0) {
        msg.events.forEach(e => addEventToFeed(e));
    }
    
    // Add policy events (large announcements)
    if (msg.policy_events && msg.policy_events.length > 0) {
        msg.policy_events.forEach(e => addEventToFeed(e, true));
    }
    
    // If inspector is open, refresh occasionally
    if (State.inspectorOpen && State.selectedAgentId !== null && msg.timestep % 10 === 0) {
        sendCommand('inspect', State.selectedAgentId);
    }
}

function handlePolicyResult(msg) {
    const btn = document.getElementById('btn-apply-policy');
    btn.innerHTML = '<i class="fas fa-magic"></i> Evaluate & Apply';
    btn.disabled = false;
    
    if (msg.success) {
        const policy = msg.data.policy;
        const resultArea = document.getElementById('policy-result-area');
        
        document.getElementById('result-policy-name').innerText = policy.policy_name;
        document.getElementById('result-source').innerText = policy.source.toUpperCase();
        document.getElementById('result-reasoning').innerText = policy.reasoning;
        
        // Global effects
        const ulGlobal = document.getElementById('result-global');
        ulGlobal.innerHTML = '';
        if (Object.keys(policy.global_effects).length === 0) {
            ulGlobal.innerHTML = '<li><span class="text-muted">No global effects</span></li>';
        } else {
            for (const [k, v] of Object.entries(policy.global_effects)) {
                ulGlobal.innerHTML += `<li><span>${k}</span> <span class="text-blue font-bold">${v}</span></li>`;
            }
        }
        
        // Agent effects
        const ulAgents = document.getElementById('result-agents');
        ulAgents.innerHTML = '';
        if (Object.keys(policy.affected_agents).length === 0) {
            ulAgents.innerHTML = '<li><span class="text-muted">No specific agent effects</span></li>';
        } else {
            for (const [type, params] of Object.entries(policy.affected_agents)) {
                const paramStrs = Object.entries(params).map(([pk, pv]) => `${pk}:${pv}`).join(', ');
                ulAgents.innerHTML += `<li><span class="text-capitalize">${type.replace('_', ' ')}</span> <span class="text-green text-sm">${paramStrs}</span></li>`;
            }
        }
        
        resultArea.classList.remove('hidden');
        
        // Clear input
        document.getElementById('policy-input').value = '';
    } else {
        alert("Failed to apply policy: " + msg.error);
    }
}

function handleComparisonResult(data) {
    document.getElementById('compare-loading').classList.add('hidden');
    const resultsArea = document.getElementById('compare-results');
    document.getElementById('btn-run-compare').disabled = false;
    
    // Set score
    const scoreCircle = document.getElementById('compare-score');
    scoreCircle.innerText = data.metrics.policy_effectiveness.toFixed(1);
    
    // Color score
    if (data.metrics.policy_effectiveness > 50) scoreCircle.style.background = 'linear-gradient(135deg, #10b981, #3b82f6)';
    else if (data.metrics.policy_effectiveness < 0) scoreCircle.style.background = 'linear-gradient(135deg, #ef4444, #f59e0b)';
    else scoreCircle.style.background = 'linear-gradient(135deg, #f59e0b, #3b82f6)';
    
    // Populate table
    const tbody = document.getElementById('compare-table-body');
    tbody.innerHTML = '';
    
    const metricsMap = [
        { key: 'gini', label: 'Gini Coefficient', isPct: true, invertGood: true },
        { key: 'poverty_rate', label: 'Poverty Rate', isPct: true, invertGood: true },
        { key: 'bankruptcy_rate', label: 'Bankruptcy Rate', isPct: true, invertGood: true },
        { key: 'median_wealth', label: 'Median Wealth (₹)', isPct: false, invertGood: false },
        { key: 'total_wealth', label: 'Total Wealth (₹)', isPct: false, invertGood: false },
        { key: 'top_10_concentration', label: 'Top 10% Wealth %', isPct: true, invertGood: true }
    ];
    
    metricsMap.forEach(m => {
        const d = data.metrics[m.key];
        if (!d) return;
        
        let aStr, bStr, deltaStr, deltaClass = '';
        
        if (m.isPct) {
            aStr = (d.a * 100).toFixed(1) + '%';
            bStr = (d.b * 100).toFixed(1) + '%';
            deltaStr = (d.delta > 0 ? '+' : '') + (d.delta * 100).toFixed(1) + '%';
        } else {
            aStr = Math.round(d.a).toLocaleString();
            bStr = Math.round(d.b).toLocaleString();
            deltaStr = (d.delta > 0 ? '+' : '') + Math.round(d.delta).toLocaleString();
        }
        
        // Color coding logic
        if (d.delta !== 0) {
            const isGood = m.invertGood ? (d.delta < 0) : (d.delta > 0);
            deltaClass = isGood ? 'delta-pos' : 'delta-neg';
        }
        
        tbody.innerHTML += `
            <tr>
                <td>${m.label}</td>
                <td>${aStr}</td>
                <td>${bStr}</td>
                <td class="font-bold ${deltaClass}">${deltaStr}</td>
            </tr>
        `;
    });
    
    resultsArea.classList.remove('hidden');
}

// ═══════════════════════════════════════════
// UI UPDATES
// ═══════════════════════════════════════════

function updateTopBar(msg) {
    document.getElementById('time-display').innerText = msg.time_label || `Timestep ${msg.timestep}`;
    
    const phaseEl = document.getElementById('phase-display');
    const phase = msg.time_of_day || "Day";
    
    let icon = "fa-sun";
    let color = "#f59e0b"; // yellow
    
    if (phase === "Dawn") { icon = "fa-sun"; color = "#f4a261"; }
    else if (phase === "Dusk") { icon = "fa-moon"; color = "#e76f51"; }
    else if (phase === "Night") { icon = "fa-moon"; color = "#a5b4fc"; }
    
    phaseEl.innerHTML = `<i class="fas ${icon}" style="color:${color}"></i> ${phase}`;
    
    // Update sky overlay color
    if (msg.sky_color) {
        document.getElementById('sky-overlay').style.backgroundColor = msg.sky_color;
    }
}

function updateSpeedButtons(speed) {
    const btns = ['pause', 'play', 'fast', 'very-fast'];
    const map = { 'pause': 'pause', 'play': 'play', 'week': 'fast', 'month': 'very-fast' };
    
    btns.forEach(b => document.getElementById(`btn-${b}`).classList.remove('active'));
    
    const activeId = map[speed];
    if (activeId) {
        document.getElementById(`btn-${activeId}`).classList.add('active');
    }
}

function updateDashboardMetrics(metrics, social_stats) {
    document.getElementById('val-gini').innerText = metrics.gini.toFixed(3);
    document.getElementById('val-poverty').innerText = (metrics.poverty_rate * 100).toFixed(1) + '%';
    document.getElementById('val-median').innerText = '₹' + Math.round(metrics.median_wealth).toLocaleString();
    document.getElementById('val-bankrupt').innerText = metrics.bankruptcy_count;
    
    if (social_stats) {
        document.getElementById('val-density').innerText = (social_stats.network_density || 0).toFixed(4);
        document.getElementById('val-trust').innerText = (social_stats.avg_trust || 0).toFixed(2);
        document.getElementById('val-active-loans').innerText = (social_stats.active_loans || 0);
        document.getElementById('val-reputation').innerText = (social_stats.avg_reputation || 0).toFixed(2);
    }
    
    // Social Mobility Bar
    if (metrics.social_mobility) {
        const up = metrics.social_mobility.upward_pct * 100;
        const stag = metrics.social_mobility.stagnant_pct * 100;
        const down = metrics.social_mobility.downward_pct * 100;
        
        document.getElementById('bar-upward').style.width = `${up}%`;
        document.getElementById('bar-upward').innerText = up > 5 ? `${up.toFixed(0)}% Up` : '';
        
        document.getElementById('bar-stagnant').style.width = `${stag}%`;
        document.getElementById('bar-stagnant').innerText = stag > 10 ? `${stag.toFixed(0)}% Stagnant` : '';
        
        document.getElementById('bar-downward').style.width = `${down}%`;
        document.getElementById('bar-downward').innerText = down > 5 ? `${down.toFixed(0)}% Down` : '';
    }
}

function updateCharts(trends) {
    if (!trends || !trends.timesteps || trends.timesteps.length === 0) return;
    
    // Labels (timesteps)
    State.charts.inequality.data.labels = trends.timesteps;
    State.charts.macro.data.labels = trends.timesteps;
    
    // Gini
    State.charts.inequality.data.datasets[0].data = trends.gini;
    State.charts.inequality.update();
    
    // Macro
    State.charts.macro.data.datasets[0].data = trends.poverty_rate.map(v => v * 100);
    State.charts.macro.data.datasets[1].data = trends.median_wealth.map(v => v / 1000); // k₹
    State.charts.macro.update();
}

function appendChartData(msg) {
    const ts = msg.timestep;
    const m = msg.metrics;
    
    // Helper to push and shift
    const pushShift = (chart, dsIndex, val, maxLen = 60) => {
        const ds = chart.data.datasets[dsIndex].data;
        ds.push(val);
        if (ds.length > maxLen) ds.shift();
    };
    
    // Gini
    const labels1 = State.charts.inequality.data.labels;
    labels1.push(ts);
    if (labels1.length > 60) labels1.shift();
    pushShift(State.charts.inequality, 0, m.gini);
    State.charts.inequality.update();
    
    // Macro
    const labels2 = State.charts.macro.data.labels;
    labels2.push(ts);
    if (labels2.length > 60) labels2.shift();
    pushShift(State.charts.macro, 0, m.poverty_rate * 100);
    pushShift(State.charts.macro, 1, m.median_wealth / 1000);
    State.charts.macro.update();
}

function addEventToFeed(text, isPolicy = false) {
    const feed = document.getElementById('feed-content');
    
    // Check if we already have this exact text to prevent spam
    const lastElements = Array.from(feed.children).slice(-3);
    if (lastElements.some(el => el.innerText === text)) {
        return;
    }
    
    const el = document.createElement('div');
    el.className = 'event-item' + (isPolicy ? ' policy' : '');
    
    // Format text: Bold agent IDs
    let formattedHtml = text;
    // Replace Agent_042 with bold span
    formattedHtml = formattedHtml.replace(/(Agent_\d+)/g, '<strong>$1</strong>');
    
    el.innerHTML = formattedHtml;
    
    feed.appendChild(el);
    
    // Keep max 15 items
    while (feed.children.length > 15) {
        feed.removeChild(feed.firstChild);
    }
    
    // Scroll to bottom
    feed.scrollTop = feed.scrollHeight;
}

function updateActivePoliciesList(policies) {
    const list = document.getElementById('active-policies-list');
    
    if (!policies || policies.length === 0) {
        list.innerHTML = '<div class="empty-state">No active policies</div>';
        return;
    }
    
    list.innerHTML = '';
    policies.forEach(p => {
        list.innerHTML += `
            <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 4px; border-left: 3px solid #8b5cf6;">
                <div style="font-weight: 600; color: #fff; font-size: 14px;">${p.name}</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Expires: ${p.duration < 0 ? 'Never' : 'T+' + (p.expires_at - State.timestep)}</div>
            </div>
        `;
    });
}

// ═══════════════════════════════════════════
// AGENT INSPECTOR
// ═══════════════════════════════════════════

function handleCanvasClick(x, y) {
    // Find closest agent within 20px radius
    let closestId = null;
    let closestDist = 20 * 20; // Squared
    
    for (const [id, agent] of State.agents.entries()) {
        const dx = agent.x - x;
        const dy = agent.y - y;
        const distSq = dx*dx + dy*dy;
        
        if (distSq < closestDist) {
            closestDist = distSq;
            closestId = id;
        }
    }
    
    if (closestId !== null) {
        // Select and open inspector tab
        State.selectedAgentId = closestId;
        State.inspectorOpen = true;
        
        // Force tab switch
        document.querySelector('.tab-btn[data-target="inspector"]').click();
        
        // Show loading state or direct cached info
        document.getElementById('insp-name').innerText = `Agent #${closestId.toString().padStart(3, '0')}`;
        document.getElementById('inspector-empty').classList.add('hidden');
        document.getElementById('inspector-data').classList.remove('hidden');
        
        // Request full data from backend
        sendCommand('inspect', closestId);
    }
}

function updateInspectorPanel(data) {
    // Populate agent header
    document.getElementById('insp-name').innerText = `Agent #${data.id.toString().padStart(3, '0')}`;
    document.getElementById('insp-type').innerText = data.type_name;
    
    // Avatar styling based on tier
    const avatar = document.getElementById('insp-avatar');
    avatar.innerText = EMOTION_ICONS[data.emotion] || "👤";
    
    // Finances
    document.getElementById('insp-wealth').innerText = '₹' + Math.round(data.wealth).toLocaleString();
    document.getElementById('insp-income').innerText = '₹' + Math.round(data.monthly_income).toLocaleString() + '/mo';
    document.getElementById('insp-debt').innerText = '₹' + Math.round(data.debt).toLocaleString();
    
    // Map tier int to string (0=Bankrupt, 1=Poor, 2=Low, 3=Middle, 4=Rich)
    const tiers = ["Bankrupt", "Poor", "Low", "Middle", "Rich"];
    const tierName = (data.wealth_tier >= 0 && data.wealth_tier < tiers.length) ? tiers[data.wealth_tier] : "Unknown";
    document.getElementById('insp-tier').innerText = tierName;
    
    // Update Action Probabilities Chart
    if (data.action_probs_raw && data.action_probs_raw.length === 4) {
        State.charts.actions.data.datasets[0].data = data.action_probs_raw.map(p => p * 100);
        State.charts.actions.update();
    }
    
    // Top Features
    const featList = document.getElementById('insp-features');
    featList.innerHTML = '';
    if (data.top_influential_features) {
        data.top_influential_features.forEach(f => {
            const valStr = f.value.toFixed(2);
            featList.innerHTML += `<li><span>${f.name}</span> <span class="font-bold text-blue">${valStr}</span></li>`;
        });
    }
    
    // Timeline
    const timeline = document.getElementById('insp-timeline');
    timeline.innerHTML = '';
    if (data.life_events && data.life_events.length > 0) {
        // Reverse to show newest first
        const recent = [...data.life_events].reverse().slice(0, 5);
        recent.forEach(e => {
            timeline.innerHTML += `
                <li>
                    <span class="time">T=${e.timestep}</span>
                    ${e.description}
                </li>
            `;
        });
    } else {
        timeline.innerHTML = '<li class="text-muted italic">No notable events yet</li>';
    }
}

// ═══════════════════════════════════════════
// CANVAS RENDERING
// ═══════════════════════════════════════════

function renderLoop(timestamp) {
    // Calculate delta time for smooth lerping
    const dt = (timestamp - State.lastFrameTime) / 1000;
    State.lastFrameTime = timestamp;
    
    if (State.ctx) {
        // 1. Clear background
        State.ctx.fillStyle = '#0a0a0a';
        State.ctx.fillRect(0, 0, State.width, State.height);
        
        // 2. Draw static city (zones, roads, buildings)
        drawCity();
        
        // 3. Update and draw agents (with lerping)
        drawAgents(dt);
        
        // 4. Draw selection highlight
        if (State.selectedAgentId !== null && State.smoothAgents.has(State.selectedAgentId)) {
            const s = State.smoothAgents.get(State.selectedAgentId);
            State.ctx.beginPath();
            State.ctx.arc(s.x, s.y, 15, 0, Math.PI * 2);
            State.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            State.ctx.lineWidth = 2;
            State.ctx.setLineDash([5, 5]);
            State.ctx.stroke();
            State.ctx.setLineDash([]);
        }
    }
    
    // Loop
    requestAnimationFrame(renderLoop);
}

function drawCity() {
    if (!State.cityData) return;
    const ctx = State.ctx;
    
    // Draw Zones
    if (State.cityData.zones) {
        for (const [name, zone] of Object.entries(State.cityData.zones)) {
            ctx.fillStyle = zone.color;
            ctx.fillRect(zone.x1, zone.y1, zone.x2 - zone.x1, zone.y2 - zone.y1);
            
            // Draw zone label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.font = 'bold 24px Outfit';
            ctx.fillText(zone.label.toUpperCase(), zone.x1 + 10, zone.y1 + 30);
        }
    }
    
    // Draw Roads
    if (State.cityData.roads && State.cityData.roads.segments) {
        const rc = State.cityData.roads;
        
        // Base road
        ctx.beginPath();
        for (const seg of rc.segments) {
            ctx.moveTo(seg.x1, seg.y1);
            ctx.lineTo(seg.x2, seg.y2);
        }
        ctx.strokeStyle = rc.color;
        ctx.lineWidth = rc.width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
        
        // Center line
        ctx.beginPath();
        for (const seg of rc.segments) {
            ctx.moveTo(seg.x1, seg.y1);
            ctx.lineTo(seg.x2, seg.y2);
        }
        ctx.strokeStyle = rc.line_color;
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 10]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    // Draw Buildings
    if (State.cityData.buildings) {
        for (const b of State.cityData.buildings) {
            // Main structure
            ctx.fillStyle = b.color;
            ctx.fillRect(b.x, b.y, b.width, b.height);
            
            // Roof / Top border
            ctx.fillStyle = b.roof_color;
            ctx.fillRect(b.x, b.y, b.width, 4);
            
            // Windows
            if (b.has_windows) {
                // Determine if windows should be lit based on phase
                // Actually, backend passes sky_color, but we can do a simple check
                const isNight = document.getElementById('phase-display').innerText.includes("Night");
                
                ctx.fillStyle = isNight ? b.window_color : 'rgba(0,0,0,0.5)';
                
                // Simple 2x2 grid or grid based on size
                const cols = Math.floor(b.width / 15);
                const rows = Math.floor(b.height / 15);
                
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        // Skip some randomly for realistic lit windows
                        if (isNight && Math.random() < 0.3) continue;
                        
                        ctx.fillRect(b.x + 5 + (c * 15), b.y + 10 + (r * 15), 6, 8);
                    }
                }
            }
        }
    }
}

function drawAgents(dt) {
    const ctx = State.ctx;
    
    // Fast lerp multiplier (smooth movement)
    // Adjust based on speed setting
    let lerpFactor = 10.0 * dt;
    if (State.speed === 'month') lerpFactor = 1.0; // Instant jump at high speed
    
    for (const [id, agent] of State.agents.entries()) {
        const s = State.smoothAgents.get(id);
        if (!s) continue;
        
        // Lerp position toward target
        s.x += (s.targetX - s.x) * lerpFactor;
        s.y += (s.targetY - s.y) * lerpFactor;
        
        // Determine agent visual size/style based on tier
        let radius = 4;
        if (agent.wealth_tier === "rich") radius = 6;
        else if (agent.wealth_tier === "bankrupt") radius = 3;
        
        // Draw glow for rich/bankrupt
        if (agent.wealth_tier === "rich") {
            ctx.beginPath();
            ctx.arc(s.x, s.y, radius + 4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 215, 0, 0.3)'; // Gold glow
            ctx.fill();
        } else if (agent.is_bankrupt) {
            ctx.beginPath();
            ctx.arc(s.x, s.y, radius + 4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'; // Red glow
            ctx.fill();
        }
        
        // Draw Agent Body
        ctx.beginPath();
        ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = agent.body_color || '#ffffff';
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw Agent Type Name below them
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '9px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(agent.type_name, s.x, s.y + radius + 3);
        
        // Draw Emotion Icon if present and not neutral
        if (agent.emotion && agent.emotion !== 'neutral') {
            const icon = EMOTION_ICONS[agent.emotion];
            if (icon) {
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                // Float icon slightly above agent
                ctx.fillText(icon, s.x, s.y - 12);
            }
        }
    }
}

// ═══════════════════════════════════════════
// NETWORK GRAPH RENDERING
// ═══════════════════════════════════════════

let networkSimulation = null;

function renderNetworkGraph(edgesData) {
    if (!window.d3) return;
    const container = document.getElementById('network-graph-container');
    if (!container) return;
    container.innerHTML = '';
    
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    
    const nodesMap = new Map();
    edgesData.forEach(e => {
        if (!nodesMap.has(e.source)) nodesMap.set(e.source, {id: e.source});
        if (!nodesMap.has(e.target)) nodesMap.set(e.target, {id: e.target});
    });
    const nodes = Array.from(nodesMap.values());
    const links = edgesData.map(e => ({...e})); 
    
    const svg = d3.select('#network-graph-container')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`);
        
    networkSimulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.is_family ? 40 : 100))
        .force('charge', d3.forceManyBody().strength(-80))
        .force('center', d3.forceCenter(width / 2, height / 2));
        
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke-width', d => Math.max(1, d.friendship * 5))
        .attr('stroke', d => d.is_family ? '#f59e0b' : (d.has_loan ? '#ef4444' : '#10b981'))
        .attr('stroke-opacity', d => Math.max(0.2, d.trust));
        
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', 8)
        .attr('fill', '#6366f1')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
            
    node.append('title').text(d => `Agent ${d.id}`);
        
    networkSimulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
            
        node
            .attr('cx', d => Math.max(8, Math.min(width - 8, d.x)))
            .attr('cy', d => Math.max(8, Math.min(height - 8, d.y)));
    });
    
    function dragstarted(event, d) {
        if (!event.active) networkSimulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x; d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) networkSimulation.alphaTarget(0);
        d.fx = null; d.fy = null;
    }
}

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.dataset.target === 'network' && State.latestSocialGraph) {
            setTimeout(() => renderNetworkGraph(State.latestSocialGraph), 100);
        }
    });
});
