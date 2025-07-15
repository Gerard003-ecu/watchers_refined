document.addEventListener('DOMContentLoaded', async () => {
    const data = await (await fetch('pro_panel_ipu_data.json')).json();

    /* ---------- Navegación ---------- */
    const navLinks = document.querySelectorAll('.nav-link, .nav-link-mobile');
    const sections = document.querySelectorAll('.content-section');
    const mobileMenu = document.getElementById('mobile-menu');
    const mobileBtn = document.getElementById('mobile-menu-button');

    function showSection(id) {
        sections.forEach(s => s.classList.toggle('active', s.id === id));
        navLinks.forEach(l => {
            const isActive = l.getAttribute('href') === `#${id}`;
            if (l.classList.contains('nav-link')) {
                l.classList.toggle('active', isActive);
            } else {
                l.classList.toggle('bg-slate-700', isActive);
            }
        });
        mobileMenu.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    navLinks.forEach(l => l.addEventListener('click', e => {
        e.preventDefault();
        showSection(l.getAttribute('href').slice(1));
    }));
    mobileBtn.addEventListener('click', () => mobileMenu.classList.toggle('hidden'));

    /* ---------- Componentes ---------- */
    const { componentData } = data;
    const compBtns = document.querySelectorAll('.component-btn');
    const compDetails = document.getElementById('component-details');
    let effChart;

    function updateCompView(key) {
        compBtns.forEach(b => b.classList.toggle('bg-teal-500', b.dataset.component === key));
        const d = componentData[key];
        compDetails.innerHTML = d.details;
        effChart && (effChart.data.labels = [key.charAt(0).toUpperCase() + key.slice(1)],
                     effChart.data.datasets[0].data = [d.efficiency],
                     effChart.update());
    }

    function createEffChart() {
        const ctx = document.getElementById('efficiencyChart');
        if (!ctx) return;
        effChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Eficiencia (%)', data: [], backgroundColor: ['#0ea5e9'], borderWidth: 0, barThickness: 40 }] },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                scales: { x: { min: 85, max: 100, grid: { color: 'rgba(255,255,255,.1)' }, ticks: { color: '#94a3b8' } }, y: { display: false } },
                plugins: { legend: { display: false } }
            }
        });
    }

    compBtns.forEach(b => b.addEventListener('click', () => updateCompView(b.dataset.component)));

    /* ---------- Baterías ---------- */
    const { batteryData } = data;
    const batDet = document.getElementById('battery-details');
    let batChart;

    function updateBatDetails(i) {
        const d = batteryData.details[i];
        batDet.innerHTML = `<h4 class="font-bold text-lg text-slate-200 mb-2">${d.title}</h4><p class="text-sm text-slate-400">${d.text}</p>`;
    }

    function createBatChart() {
        const ctx = document.getElementById('batteryChart');
        if (!ctx) return;
        batChart = new Chart(ctx, {
            type: 'bar',
            data: batteryData,
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,.1)' }, ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#94a3b8' } }, tooltip: { mode: 'index' } },
                onClick: (_, els) => els.length && updateBatDetails(els[0].index)
            }
        });
    }

    /* ---------- Vida útil vs DoD ---------- */
    const { cycleData } = data;
    function createCycleChart() {
        const ctx = document.getElementById('cycleLifeChart');
        if (!ctx) return;
        new Chart(ctx, {
            type: 'line',
            data: cycleData,
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Profundidad de Descarga', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'Ciclos de Vida', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#94a3b8' } } }
            }
        });
    }

    /* ---------- Tooltips ---------- */
    const tooltip = document.getElementById('tooltip');
    document.querySelectorAll('.diagram-block').forEach(block => {
        block.addEventListener('mouseenter', e => {
            tooltip.textContent = e.target.dataset.tooltip || '';
            tooltip.style.display = 'block';
            const r = e.target.getBoundingClientRect();
            tooltip.style.left = `${r.left}px`;
            tooltip.style.top = `${r.top - tooltip.offsetHeight - 8}px`;
        });
        block.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    });

    /* ---------- Pistón Animation ---------- */
    const rod = document.getElementById('piston-rod');
    const head = document.getElementById('piston-head');
    const charge = document.getElementById('capacitor-charge');
    const magnetic = document.getElementById('magnetic-field');
    const sparkEl = document.getElementById('spark');
    const wave = document.getElementById('energy-wave');
    const level = document.getElementById('energy-level');
    const capState = document.getElementById('capacitor-state');
    const magState = document.getElementById('magnetic-state');

    let step = 0;
    function runAnimation() {
        const steps = [
            () => { charge.style.height = '100%'; rod.style.top = '100px'; head.style.top = '170px'; level.style.width = '30%'; capState.textContent = 'Cargando...'; magState.textContent = 'Inactivo'; },
            () => { level.style.width = '100%'; capState.textContent = 'Carga completa (1000 V)'; },
            () => { charge.style.height = '50%'; rod.style.top = '50px'; head.style.top = '80px'; magnetic.style.opacity = '1'; level.style.width = '70%'; capState.textContent = 'Descargando...'; magState.textContent = 'Campo magnético creciendo'; },
            () => { magnetic.classList.add('magnetic-pulse'); level.style.width = '40%'; magState.textContent = 'Campo magnético máximo'; },
            () => { charge.style.height = '0%'; rod.style.top = '0px'; head.style.top = '30px'; magnetic.style.opacity = '0'; sparkEl.style.opacity = '1'; wave.style.opacity = '1'; wave.style.transform = 'scale(1.5)'; level.style.width = '150%'; capState.textContent = 'Descarga completa'; magState.textContent = 'Kickback inductivo!'; },
            () => { sparkEl.style.opacity = '0'; wave.style.opacity = '0'; wave.style.transform = 'scale(0)'; level.style.width = '0%'; capState.textContent = 'Preparando nuevo ciclo...'; magState.textContent = 'Campo colapsado'; }
        ];
        steps[step]();
        step = (step + 1) % steps.length;
    }

    /* ---------- Init ---------- */
    showSection('inicio');
    createEffChart();
    updateCompView('inversor');
    createBatChart();
    updateBatDetails(0);
    createCycleChart();
    if (rod) setInterval(runAnimation, 2000);
});
