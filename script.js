document.addEventListener('DOMContentLoaded', () => {

    // =========================
    // GLOBAL STATE
    // =========================
    let musicData = [];

    const resultCard = document.getElementById('result-card');
    const loader = document.getElementById('loader');
    const predictionOutput = document.getElementById('prediction-output');

    // =========================
    // LOAD DATA
    // =========================
    Papa.parse('data/youtube_music_stats.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: function (results) {
            musicData = results.data;
            populateData(musicData);
        }
    });

    let modelMetrics = null;

    // =========================
    // LOAD MODEL METRICS
    // =========================
    fetch('outputs/model_metrics.json')
        .then(response => response.json())
        .then(metrics => {
            modelMetrics = metrics;
            const r2Tpot = document.getElementById('sys-r2-score-tpot');
            const rmseTpot = document.getElementById('sys-rmse-tpot');
            const r2Static = document.getElementById('sys-r2-score-static');
            const rmseStatic = document.getElementById('sys-rmse-static');
            
            if (metrics.tpot) {
                if (r2Tpot) r2Tpot.innerText = (metrics.tpot.r2_score * 100).toFixed(2) + '%';
                if (rmseTpot) rmseTpot.innerText = metrics.tpot.rmse.toFixed(4);
            }
            if (metrics.static) {
                if (r2Static) r2Static.innerText = (metrics.static.r2_score * 100).toFixed(2) + '%';
                if (rmseStatic) rmseStatic.innerText = metrics.static.rmse.toFixed(4);
            }
        })
        .catch(err => console.log('Could not load metrics: ', err));

    // =========================
    // MANUAL FORM SUBMIT
    // =========================
    document.getElementById("predict-form-manual").addEventListener("submit", function (e) {
        e.preventDefault();

        const views = parseFloat(document.getElementById("current-views").value);
        const likes = parseFloat(document.getElementById("current-likes").value);
        const comments = parseFloat(document.getElementById("current-comments").value);
        const daysOld = parseFloat(document.getElementById("days-old").value);

        runPredictionProcess(views, likes, comments, daysOld);
    });

    // =========================
    // MAIN PREDICTION FUNCTION
    // =========================
    async function runPredictionProcess(views, likes, comments, daysOld) {

        resultCard.style.display = 'flex';
        loader.style.display = 'flex';
        loader.innerHTML = "<p>Running TPOT AutoML Model...</p>";
        predictionOutput.style.display = 'none';

        try {
            // Feature Engineering
            const viewsPerDay = views / daysOld;
            const engagementRate = (likes + comments) / views;
            const likeViewRatio = likes / views;
            const commentViewRatio = comments / views;

            const data = {
                views: views,
                likes: likes,
                comments: comments,
                views_per_day: viewsPerDay,
                engagement_rate: engagementRate,
                like_view_ratio: likeViewRatio,
                comment_view_ratio: commentViewRatio,
                channel_influence_score: 0.7,
                lang_avg_viral_score: 0.5
            };

            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            loader.style.display = 'none';
            predictionOutput.style.display = 'block';

            if (result.success) {
                updateUI(result);
            } else {
                alert(result.error);
            }

        } catch (error) {
            alert("Backend not running! Ensure app.py is running.");
        }
    }

    // =========================
    // UPDATE UI
    // =========================
    function updateUI(result) {
        // TPOT Updates
        const tpotScore = result.tpot_prediction || 0;
        document.getElementById("result-index-tpot").innerText = tpotScore.toFixed(4);
        setResultUI("tpot", tpotScore);
        
        const tpotModelDisplay = document.getElementById("model-name-display-tpot");
        if (tpotModelDisplay) tpotModelDisplay.innerText = result.tpot_model || "Unknown";

        // Static ML Updates
        const staticScore = result.static_prediction || 0;
        document.getElementById("result-index-static").innerText = staticScore.toFixed(4);
        setResultUI("static", staticScore);
        
        const staticModelDisplay = document.getElementById("model-name-display-static");
        if (staticModelDisplay) staticModelDisplay.innerText = result.static_model || "Unknown";

        // Details using separate scores
        const viralScoreTpotEl = document.getElementById("det-viral-score-tpot");
        if (viralScoreTpotEl) viralScoreTpotEl.innerText = tpotScore.toFixed(2);
        
        const momentumTpotEl = document.getElementById("det-momentum-tpot");
        if (momentumTpotEl) {
            let momentumTpot = Math.min(1.0, tpotScore * 1.2);
            momentumTpotEl.innerText = momentumTpot.toFixed(2);
        }
        
        const viralScoreStaticEl = document.getElementById("det-viral-score-static");
        if (viralScoreStaticEl) viralScoreStaticEl.innerText = staticScore.toFixed(2);
        
        const momentumStaticEl = document.getElementById("det-momentum-static");
        if (momentumStaticEl) {
            let momentumStatic = Math.min(1.0, staticScore * 1.2);
            momentumStaticEl.innerText = momentumStatic.toFixed(2);
        }
        
        if (result.stats) {
            const engagementTpotEl = document.getElementById("det-engagement-tpot");
            if (engagementTpotEl) engagementTpotEl.innerText = (result.stats.engagement_rate * 100).toFixed(1) + "%";
            
            const engagementStaticEl = document.getElementById("det-engagement-static");
            if (engagementStaticEl) engagementStaticEl.innerText = (result.stats.engagement_rate * 100).toFixed(1) + "%";
            
            let vPercent = Math.min(100, (result.stats.views / 500000) * 100);
            let lPercent = Math.min(100, (result.stats.likes / 25000) * 100);
            let cPercent = Math.min(100, (result.stats.comments / 2000) * 100);
            
            const barV = document.getElementById("bar-v-song");
            if (barV) barV.style.width = vPercent + "%";
            
            const barL = document.getElementById("bar-l-song");
            if (barL) barL.style.width = lPercent + "%";
            
            const barC = document.getElementById("bar-c-song");
            if (barC) barC.style.width = cPercent + "%";
        }
        
        // Dynamic Model Selection
        if (modelMetrics) {
            const tpotR2 = modelMetrics.tpot ? modelMetrics.tpot.r2_score : 0;
            const staticR2 = modelMetrics.static ? modelMetrics.static.r2_score : 0;
            
            const badgeTpot = document.getElementById("badge-tpot");
            const badgeStatic = document.getElementById("badge-static");
            const colTpot = document.getElementById("col-tpot");
            const colStatic = document.getElementById("col-static");
            
            if (badgeTpot) badgeTpot.innerHTML = "";
            if (badgeStatic) badgeStatic.innerHTML = "";
            if (colTpot) colTpot.classList.remove("dimmed-column");
            if (colStatic) colStatic.classList.remove("dimmed-column");
            
            if (tpotR2 >= staticR2) {
                if (badgeTpot) badgeTpot.innerHTML = `<div class="winner-badge"><i class="fa-solid fa-crown"></i> SELECTED MODEL (Acc: ${(tpotR2*100).toFixed(1)}%)</div>`;
                if (colStatic) colStatic.classList.add("dimmed-column");
            } else {
                if (badgeStatic) badgeStatic.innerHTML = `<div class="winner-badge"><i class="fa-solid fa-crown"></i> SELECTED MODEL (Acc: ${(staticR2*100).toFixed(1)}%)</div>`;
                if (colTpot) colTpot.classList.add("dimmed-column");
            }
        }
    }

    function setResultUI(prefix, score) {
        let emoji = "💤";
        let text = "LOW VIRALITY";
        
        if (score >= 0.7) {
            emoji = "🔥";
            text = "MEGA VIRAL";
        } else if (score >= 0.5) {
            emoji = "🚀";
            text = "VIRAL";
        } else if (score >= 0.3) {
            emoji = "📈";
            text = "TRENDING";
        }
        
        const emojiEl = document.getElementById(`result-emoji-${prefix}`);
        const textEl = document.getElementById(`result-text-${prefix}`);
        
        if (emojiEl) emojiEl.innerText = emoji;
        if (textEl) textEl.innerText = text;
    }

    // =========================
    // NAVIGATION LOGIC
    // =========================
    const navItems = document.querySelectorAll('.nav-item');
    const viewSections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all nav items and sections
            navItems.forEach(nav => nav.classList.remove('active'));
            viewSections.forEach(section => section.classList.remove('active'));
            
            // Add active class to clicked item and corresponding section
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
            
            // Update page title
            const titleMap = {
                'dashboard': ['Dashboard Overview', 'Professional Music Trend Analysis & Viral Prediction Platform'],
                'prediction-system': ['Prediction System', 'How Our AI Model Works'],
                'predict-song': ['Predict New Song', 'Run Custom Data Through The Model'],
                'visualizations': ['Visualizations', 'Deep Dive Data Analysis'],
                'data-explorer': ['Data Explorer', 'Browse and Filter Dataset']
            };
            
            if (titleMap[targetId]) {
                document.getElementById('page-title').innerText = titleMap[targetId][0];
                document.getElementById('page-subtitle').innerText = titleMap[targetId][1];
            }
        });
    });

    // =========================
    // VISUALIZATION TABS LOGIC
    // =========================
    const vizTabs = document.querySelectorAll('.viz-tab');
    const vizContents = document.querySelectorAll('.viz-content');

    vizTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            vizTabs.forEach(t => t.classList.remove('active'));
            vizContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const targetViz = tab.getAttribute('data-viz');
            document.getElementById('viz-' + targetViz).classList.add('active');
        });
    });

    // =========================
    // PREDICTION MODE TOGGLE
    // =========================
    const modeRadios = document.querySelectorAll('input[name="pred-mode"]');
    const manualForm = document.getElementById('predict-form-manual');
    const autoForm = document.getElementById('predict-form-auto');

    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'manual') {
                manualForm.style.display = 'block';
                autoForm.style.display = 'none';
            } else {
                manualForm.style.display = 'none';
                autoForm.style.display = 'block';
            }
        });
    });

    // =========================
    // AUTO PREDICT METHOD TOGGLE
    // =========================
    const autoMethod = document.getElementById('auto-method');
    const dbSelectGroup = document.getElementById('db-select-group');
    const urlInputGroup = document.getElementById('url-input-group');

    if(autoMethod) {
        autoMethod.addEventListener('change', (e) => {
            if (e.target.value === 'db') {
                dbSelectGroup.style.display = 'block';
                urlInputGroup.style.display = 'none';
            } else {
                dbSelectGroup.style.display = 'none';
                urlInputGroup.style.display = 'block';
            }
        });
    }

    // =========================
    // AUTO FORM SUBMIT
    // =========================
    const autoFormEl = document.getElementById("predict-form-auto");
    if(autoFormEl) {
        autoFormEl.addEventListener("submit", function (e) {
            e.preventDefault();
            const method = document.getElementById('auto-method').value;
            
            if (method === 'db') {
                const selectIndex = document.getElementById('db-song-select').value;
                if(selectIndex === "") {
                    alert("Please select a song from the database.");
                    return;
                }
                const song = musicData[selectIndex];
                const daysOld = song.days_since_release || 7; 
                runPredictionProcess(song.views, song.likes, song.comments, daysOld);
            } else {
                // Fetch dynamic URL data
                const urlInput = document.getElementById('youtube-url');
                const url = urlInput ? urlInput.value : '';
                if (!url) {
                    alert("Please enter a valid YouTube URL.");
                    return;
                }
                
                resultCard.style.display = 'flex';
                loader.style.display = 'flex';
                loader.innerHTML = "<p>Analyzing URL dynamically... This may take a moment.</p>";
                predictionOutput.style.display = 'none';

                fetch("/api/analyze_url", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ url: url })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        if (data.warning) {
                            console.warn("Using simulated data:", data.warning);
                        }
                        runPredictionProcess(data.views, data.likes, data.comments, data.days_old);
                    } else {
                        alert("Failed to analyze URL: " + data.error);
                        loader.style.display = 'none';
                    }
                })
                .catch(error => {
                    alert("Error reaching backend URL analyzer.");
                    loader.style.display = 'none';
                });
            }
        });
    }

    // =========================
    // POPULATE DATA EXPLORER & DROPDOWN
    // =========================
    function populateData(data) {
        const tbody = document.getElementById('data-table-body');
        const songSelect = document.getElementById('db-song-select');
        
        if(!tbody || !songSelect) return;
        
        tbody.innerHTML = '';
        songSelect.innerHTML = '<option value="">Choose a song...</option>';
        
        // Take first 100 rows to prevent lagging
        const displayData = data.slice(0, 100);
        
        displayData.forEach((row, index) => {
            // Populate table
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.title || 'Unknown'}</td>
                <td>${row.channel_name || 'Unknown'}</td>
                <td>${row.language || 'Unknown'}</td>
                <td>${(row.views || 0).toLocaleString()}</td>
                <td>${(row.virality_index || 0).toFixed(4)}</td>
                <td><span class="badge badge-${(row.virality_index >= 0.5) ? 'viral' : 'trending'}">${(row.virality_index >= 0.5) ? 'VIRAL' : 'NORMAL'}</span></td>
            `;
            tbody.appendChild(tr);
            
            // Populate dropdown
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${row.title || 'Unknown'} - ${row.channel_name || 'Unknown'}`;
            songSelect.appendChild(option);
        });
    }

});