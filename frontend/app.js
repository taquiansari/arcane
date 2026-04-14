/* ═══════════════════════════════════════════════════════════
   ARCANE — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

const API_BASE = '/api/v1';
const WS_BASE = `ws://${location.host}`;

// ── State ────────────────────────────────────────────────────
const state = {
    currentSessionId: null,
    sessions: [],
    ws: null,
    isResearching: false,
};

// ── DOM References ───────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebarToggle'),
    sessionList: $('#sessionList'),
    newResearchBtn: $('#newResearchBtn'),

    heroSection: $('#heroSection'),
    researchView: $('#researchView'),

    searchInput: $('#searchInput'),
    searchBtn: $('#searchBtn'),
    humanReview: $('#humanReview'),
    maxRevisions: $('#maxRevisions'),

    researchQuery: $('#researchQuery'),
    statusBadge: $('#statusBadge'),
    revisionBadge: $('#revisionBadge'),
    revisionCount: $('#revisionCount'),

    pipelineTracker: $('#pipelineTracker'),
    logEntries: $('#logEntries'),
    clearLog: $('#clearLog'),

    reportContainer: $('#reportContainer'),
    reportContent: $('#reportContent'),
    scoreValue: $('#scoreValue'),
    copyReportBtn: $('#copyReportBtn'),
    downloadReportBtn: $('#downloadReportBtn'),

    feedbackPanel: $('#feedbackPanel'),
    feedbackInput: $('#feedbackInput'),
    sendFeedbackBtn: $('#sendFeedbackBtn'),
    approveDraftBtn: $('#approveDraftBtn'),

    statusDot: $('#statusDot'),
    statusText: $('#statusText'),

    backBtn: $('#backBtn'),
};

// ── Initialization ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkHealth();
    loadSessions();
    autoResizeTextarea();
});

function initEventListeners() {
    // Search
    dom.searchBtn.addEventListener('click', startResearch);
    dom.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            startResearch();
        }
    });

    // Auto-resize textarea
    dom.searchInput.addEventListener('input', autoResizeTextarea);

    // Suggestions
    $$('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            dom.searchInput.value = chip.dataset.query;
            autoResizeTextarea();
            dom.searchInput.focus();
        });
    });

    // New research
    dom.newResearchBtn.addEventListener('click', showHero);

    // Back button
    dom.backBtn.addEventListener('click', showHero);

    // Sidebar toggle
    dom.sidebarToggle.addEventListener('click', () => {
        dom.sidebar.classList.toggle('collapsed');
        dom.sidebar.classList.toggle('open');
    });

    // Log clear
    dom.clearLog.addEventListener('click', () => {
        dom.logEntries.innerHTML = '';
    });

    // Report actions
    dom.copyReportBtn.addEventListener('click', copyReport);
    dom.downloadReportBtn.addEventListener('click', downloadReport);

    // Feedback
    dom.sendFeedbackBtn.addEventListener('click', sendFeedback);
    dom.approveDraftBtn.addEventListener('click', approveDraft);
}

function autoResizeTextarea() {
    dom.searchInput.style.height = 'auto';
    dom.searchInput.style.height = Math.min(dom.searchInput.scrollHeight, 120) + 'px';
}

// ── API Helpers ──────────────────────────────────────────────
async function api(method, path, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    try {
        const res = await fetch(`${API_BASE}${path}`, opts);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || err.error || `HTTP ${res.status}`);
        }
        return await res.json();
    } catch (err) {
        console.error(`API ${method} ${path} failed:`, err);
        throw err;
    }
}

// ── Health Check ─────────────────────────────────────────────
async function checkHealth() {
    try {
        const data = await api('GET', '/health');
        dom.statusDot.className = 'status-dot ' + (data.redis_connected ? 'connected' : 'disconnected');
        dom.statusText.textContent = data.redis_connected
            ? `Redis connected · v${data.version}`
            : 'Redis disconnected';
    } catch {
        dom.statusDot.className = 'status-dot disconnected';
        dom.statusText.textContent = 'API unreachable';
    }
}

// ── Sessions ─────────────────────────────────────────────────
async function loadSessions() {
    try {
        const data = await api('GET', '/sessions');
        state.sessions = data.sessions || [];
        renderSessionList();
    } catch {
        // Silently fail — sessions will be empty
    }
}

function renderSessionList() {
    if (!state.sessions.length) {
        dom.sessionList.innerHTML = `
            <div class="session-empty">
                <p>No research sessions yet.</p>
                <p class="muted">Start one above!</p>
            </div>`;
        return;
    }

    dom.sessionList.innerHTML = state.sessions.map(s => {
        const isActive = s.session_id === state.currentSessionId;
        const statusClass = getStatusClass(s.status);
        const time = formatRelativeTime(s.updated_at);

        return `
            <div class="session-item ${isActive ? 'active' : ''}"
                 data-session-id="${s.session_id}"
                 onclick="loadSession('${s.session_id}')">
                <div class="session-query">${escapeHtml(s.query)}</div>
                <div class="session-status">
                    <span class="dot ${statusClass}"></span>
                    <span>${s.status}</span>
                    <span>· ${time}</span>
                </div>
            </div>`;
    }).join('');
}

// ── Start Research ───────────────────────────────────────────
async function startResearch() {
    const query = dom.searchInput.value.trim();
    if (!query || state.isResearching) return;

    state.isResearching = true;
    dom.searchBtn.disabled = true;

    try {
        const data = await api('POST', '/research', {
            query,
            max_revisions: parseInt(dom.maxRevisions.value),
            human_review: dom.humanReview.checked,
        });

        state.currentSessionId = data.session_id;
        dom.searchInput.value = '';
        autoResizeTextarea();

        showResearchView(query);
        connectWebSocket(data.session_id);
        addLogEntry('info', `Research started: ${data.session_id.slice(0, 8)}...`);

        // Reload sessions
        setTimeout(loadSessions, 1000);

    } catch (err) {
        showToast(`Failed to start research: ${err.message}`, 'error');
    } finally {
        state.isResearching = false;
        dom.searchBtn.disabled = false;
    }
}

// ── Load Existing Session ────────────────────────────────────
async function loadSession(sessionId) {
    try {
        const data = await api('GET', `/research/${sessionId}`);
        state.currentSessionId = sessionId;

        showResearchView(data.query);
        updateStatus(data.status);

        if (data.report) {
            showReport(data.report, data.critique_score, data.revision_count);
        }

        if (data.status === 'running' || data.status === 'started') {
            connectWebSocket(sessionId);
        }

        renderSessionList();

    } catch (err) {
        showToast(`Failed to load session: ${err.message}`, 'error');
    }
}

// Make loadSession available globally for onclick handlers
window.loadSession = loadSession;

// ── WebSocket ────────────────────────────────────────────────
function connectWebSocket(sessionId) {
    if (state.ws) {
        state.ws.close();
    }

    const ws = new WebSocket(`${WS_BASE}/ws/research/${sessionId}`);
    state.ws = ws;

    ws.onopen = () => {
        addLogEntry('success', 'Connected to research stream');
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleWSEvent(msg);
        } catch (e) {
            console.error('WS parse error:', e);
        }
    };

    ws.onclose = () => {
        addLogEntry('warning', 'Stream disconnected');
        // Try to reload final state
        if (state.currentSessionId) {
            setTimeout(() => loadSession(state.currentSessionId), 2000);
        }
    };

    ws.onerror = (err) => {
        addLogEntry('error', 'WebSocket error');
        console.error('WS error:', err);
    };
}

function handleWSEvent(msg) {
    const { type, data } = msg;

    switch (type) {
        case 'status':
            updateStatus(data.stage || data.status || '');
            addLogEntry('info', data.message || `Stage: ${data.stage}`);
            updatePipeline(data.stage);
            break;

        case 'progress':
            addLogEntry('info', `Searching: "${(data.query || '').slice(0, 60)}" — ${data.results_count || 0} results`);
            break;

        case 'draft':
            addLogEntry('info', `Draft revision ${data.revision || '?'} generated`);
            dom.revisionBadge.style.display = 'flex';
            dom.revisionCount.textContent = data.revision || '?';
            break;

        case 'critique':
            const score = data.score != null ? (data.score * 100).toFixed(0) + '%' : '?';
            addLogEntry(data.score >= 0.8 ? 'success' : 'warning',
                `Critique score: ${score}`);
            break;

        case 'final':
            addLogEntry('success', 'Research complete!');
            showReport(data.report, data.critique_score, data.revision_count);
            updateStatus('complete');
            updatePipeline('complete');
            loadSessions();
            break;

        case 'error':
            addLogEntry('error', data.message || 'An error occurred');
            updateStatus('failed');
            break;

        default:
            addLogEntry('info', JSON.stringify(data).slice(0, 100));
    }
}

// ── Pipeline Tracker ─────────────────────────────────────────
const STAGE_MAP = {
    'created': 'plan',
    'started': 'plan',
    'starting': 'plan',
    'running': 'plan',
    'planning_complete': 'queries',
    'planning_fallback': 'queries',
    'queries_generated': 'retrieve',
    'queries_fallback': 'retrieve',
    'retrieval_in_progress': 'retrieve',
    'retrieval_complete': 'retrieve',
    'retrieval_error': 'retrieve',
    'analysis_complete': 'synthesize',
    'synthesis_complete': 'critique',
    'synthesis_error': 'critique',
    'critique_complete': 'critique',
    'critique_error': 'critique',
    'feedback_received': 'critique',
    'complete': 'complete',
    'failed': 'complete',
    'cancelled': 'complete',
};

const STEP_ORDER = ['plan', 'queries', 'retrieve', 'synthesize', 'critique', 'complete'];

function updatePipeline(stage) {
    const mappedStep = STAGE_MAP[stage] || stage;
    const activeIndex = STEP_ORDER.indexOf(mappedStep);

    $$('.pipeline-step').forEach((el, i) => {
        el.classList.remove('active', 'done');
        if (i < activeIndex) {
            el.classList.add('done');
        } else if (i === activeIndex) {
            el.classList.add('active');
        }
    });
}

// ── Report Display ───────────────────────────────────────────
function showReport(report, score, revisions) {
    if (!report) return;

    dom.reportContainer.classList.remove('hidden');
    dom.reportContent.innerHTML = renderMarkdown(report);

    if (score != null) {
        dom.scoreValue.textContent = (score * 100).toFixed(0) + '%';
        dom.scoreValue.style.color = score >= 0.8 ? 'var(--success)' : 'var(--warning)';
    }

    if (revisions != null) {
        dom.revisionBadge.style.display = 'flex';
        dom.revisionCount.textContent = revisions;
    }

    // Scroll to report
    dom.reportContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderMarkdown(md) {
    if (!md) return '';

    // Basic Markdown → HTML conversion
    let html = md
        // Code blocks
        .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Headers
        .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Bold & Italic
        .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Blockquotes
        .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
        // Horizontal rules
        .replace(/^---$/gm, '<hr>')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
        // Unordered lists
        .replace(/^[*-] (.+)$/gm, '<li>$1</li>')
        // Ordered lists
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Paragraphs (double newlines)
        .replace(/\n\n/g, '</p><p>')
        // Single newlines in context
        .replace(/\n/g, '<br>');

    // Wrap list items
    html = html.replace(/(<li>.*?<\/li>)/gs, (match) => {
        if (!match.startsWith('<ul>') && !match.startsWith('<ol>')) {
            return '<ul>' + match + '</ul>';
        }
        return match;
    });

    // Merge adjacent ul tags
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Wrap in paragraph tags
    html = '<p>' + html + '</p>';
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[1-4]>)/g, '$1');
    html = html.replace(/(<\/h[1-4]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<blockquote>)/g, '$1');
    html = html.replace(/(<\/blockquote>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<hr>)/g, '$1');
    html = html.replace(/(<hr>)\s*<\/p>/g, '$1');

    return html;
}

// ── Report Actions ───────────────────────────────────────────
function copyReport() {
    const text = dom.reportContent.innerText;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Report copied to clipboard');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

function downloadReport() {
    const text = dom.reportContent.innerText;
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `arcane-report-${state.currentSessionId?.slice(0, 8) || 'unknown'}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Report downloaded');
}

// ── Feedback (HITL) ──────────────────────────────────────────
async function sendFeedback() {
    const feedback = dom.feedbackInput.value.trim();
    if (!feedback || !state.currentSessionId) return;

    try {
        await api('POST', `/research/${state.currentSessionId}/feedback`, {
            feedback,
            approve: false,
        });
        dom.feedbackInput.value = '';
        addLogEntry('success', 'Feedback submitted — revision in progress');
        showToast('Feedback submitted');
    } catch (err) {
        showToast(`Failed: ${err.message}`, 'error');
    }
}

async function approveDraft() {
    if (!state.currentSessionId) return;

    try {
        await api('POST', `/research/${state.currentSessionId}/feedback`, {
            feedback: 'Approved by reviewer',
            approve: true,
        });
        dom.feedbackPanel.classList.add('hidden');
        addLogEntry('success', 'Draft approved — finalizing report');
        showToast('Draft approved');
    } catch (err) {
        showToast(`Failed: ${err.message}`, 'error');
    }
}

// ── View Management ──────────────────────────────────────────
function showHero() {
    dom.heroSection.classList.remove('hidden');
    dom.researchView.classList.add('hidden');
    state.currentSessionId = null;
    renderSessionList();

    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
}

function showResearchView(query) {
    dom.heroSection.classList.add('hidden');
    dom.researchView.classList.remove('hidden');
    dom.researchQuery.textContent = query;

    // Reset view state
    dom.reportContainer.classList.add('hidden');
    dom.feedbackPanel.classList.add('hidden');
    dom.logEntries.innerHTML = '';
    dom.revisionBadge.style.display = 'none';
    dom.scoreValue.textContent = '—';

    // Reset pipeline
    $$('.pipeline-step').forEach(el => {
        el.classList.remove('active', 'done');
    });

    updateStatus('starting');
    updatePipeline('starting');
}

function updateStatus(status) {
    const badge = dom.statusBadge;
    badge.textContent = formatStatus(status);
    badge.className = 'badge ' + getStatusClass(status);
}

// ── Activity Log ─────────────────────────────────────────────
function addLogEntry(level, message) {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-msg">${escapeHtml(message)}</span>
    `;

    dom.logEntries.appendChild(entry);
    dom.logEntries.scrollTop = dom.logEntries.scrollHeight;
}

// ── Toast Notifications ──────────────────────────────────────
function showToast(message, type = 'info') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    if (type === 'error') {
        toast.style.borderColor = 'var(--error)';
    }
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('leaving');
        setTimeout(() => toast.remove(), 200);
    }, 3000);
}

// ── Utility Functions ────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getStatusClass(status) {
    if (!status) return 'created';
    status = status.toLowerCase();
    if (['complete', 'approved'].includes(status)) return 'complete';
    if (['failed', 'error', 'cancelled'].includes(status)) return 'failed';
    if (['running', 'started', 'starting'].includes(status)) return 'running';
    if (status.includes('progress') || status.includes('complete')) return 'running';
    return 'created';
}

function formatStatus(status) {
    if (!status) return 'Unknown';
    return status.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatRelativeTime(timestamp) {
    const now = Date.now() / 1000;
    const diff = now - timestamp;

    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

// ── Periodic Refresh ─────────────────────────────────────────
// Refresh sessions and health every 30 seconds
setInterval(() => {
    checkHealth();
    loadSessions();
}, 30000);

// Polling fallback for non-WebSocket status updates
setInterval(async () => {
    if (!state.currentSessionId) return;

    // Only poll if WebSocket is not connected
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;

    try {
        const data = await api('GET', `/research/${state.currentSessionId}`);
        updateStatus(data.status);
        updatePipeline(data.status);

        if (data.report && data.status === 'complete') {
            showReport(data.report, data.critique_score, data.revision_count);
        }
    } catch {
        // Silently fail
    }
}, 5000);
