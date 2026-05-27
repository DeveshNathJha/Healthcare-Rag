import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  // Conversational RAG state
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [checkRelevance, setCheckRelevance] = useState(false);

  // Ingestion & Registry state
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);

  // Stats / Monitoring state
  const [stats, setStats] = useState({
    upload_count: 0,
    index_size_mb: 0,
    index_exists: false,
    token_budget: {
      total_queries: 0,
      api_calls_made: 0,
      cache_hits: 0,
      cache_hit_rate_pct: 0,
      total_input_tokens: 0,
      total_output_tokens: 0,
      model_8b_calls: 0,
      model_70b_calls: 0,
      estimated_cost_usd: 0,
    },
    prompt_cache: {
      cached_queries: 0,
      total_cache_hits: 0,
    },
  });

  // UI state
  const [editingMsgIndex, setEditingMsgIndex] = useState(null);
  const [editText, setEditText] = useState("");
  const [expandedEvalIndex, setExpandedEvalIndex] = useState(null);
  const chatEndRef = useRef(null);

  // Fetch initial files and stats
  useEffect(() => {
    fetchFiles();
    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Dynamic stats polling
    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const fetchFiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/list-files`);
      const data = await res.json();
      setFiles(data.files || []);
    } catch (err) {
      console.error("Failed to fetch files:", err);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      const data = await res.json();
      if (data && data.token_budget) {
        setStats(data);
      }
    } catch (err) {
      console.error("Failed to fetch system stats:", err);
    }
  };

  // Upload status textual sub-stages
  const [uploadStage, setUploadStage] = useState("");

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(5);
    setUploadStage("Uploading medical file to server...");
    setUploadResult(null);

    // Fluid progress simulation (keeps user wowed and informed)
    const startTime = Date.now();
    const intervalId = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 95) {
          // Slow dynamic messages for heavy OCR files
          const elapsed = (Date.now() - startTime) / 1000;
          if (elapsed > 12) {
            setUploadStage("Running CPU OCR OCR & OpenCV preprocessing layers...");
          } else if (elapsed > 6) {
            setUploadStage("Parsing clinical text structures & chunk splitters...");
          }
          return 95;
        }
        
        // Dynamic increments
        const increment = prev < 50 ? 8 : (prev < 80 ? 4 : 1);
        
        // Update stage based on progress threshold
        if (prev < 30) {
          setUploadStage("Reading document streams...");
        } else if (prev < 60) {
          setUploadStage("Segmenting child/parent medical records...");
        } else {
          setUploadStage("Generating FAISS sentence vector embeddings...");
        }
        
        return prev + increment;
      });
    }, 350);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      clearInterval(intervalId);
      setUploadProgress(100);
      setUploadStage("Ingestion complete!");

      const data = await res.json();

      if (res.ok) {
        setUploadResult({
          success: true,
          message: data.message,
          pages: data.pages_processed,
          chars: data.total_chars,
          time: data.ingestion_time_sec,
        });
        fetchFiles();
        fetchStats();
      } else {
        setUploadResult({
          success: false,
          message: data.detail || "Upload failed.",
        });
      }
    } catch (err) {
      clearInterval(intervalId);
      setUploadProgress(0);
      setUploadStage("Ingestion failed.");
      setUploadResult({
        success: false,
        message: "Failed to connect to ingestion server.",
      });
    } finally {
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
        setUploadStage("");
      }, 4000);
    }
  };

  const handleDeleteDocument = async (filename) => {
    if (!confirm(`Are you sure you want to delete "${filename}"? This will remove it from the upload registry.`)) return;

    try {
      const res = await fetch(`${API_BASE}/delete-document?filename=${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      alert(data.message || data.warning || "File deleted.");
      fetchFiles();
      fetchStats();
      if (selectedFile === filename) setSelectedFile("");
    } catch (err) {
      console.error("Failed to delete file:", err);
    }
  };

  const handleClearDatabase = async () => {
    if (!confirm("CRITICAL WARNING: Are you sure you want to completely reset the RAG database? This will permanently delete all uploaded files, wipe the vector index, empty the cache, and reset all observability counters!")) return;

    try {
      const res = await fetch(`${API_BASE}/clear-database`, {
        method: "POST"
      });
      const data = await res.json();
      if (res.ok) {
        alert("Success: The database and vector index have been completely reset.");
        setMessages([]);
        setFiles([]);
        setSelectedFile("");
        fetchFiles();
        fetchStats();
      } else {
        alert(`Error: ${data.detail || "Failed to reset database."}`);
      }
    } catch (err) {
      console.error("Failed to clear database:", err);
      alert("Failed to connect to database reset endpoint.");
    }
  };

  const handleQuery = async (e, customText = null, isEdit = false, editIndex = null) => {
    if (e) e.preventDefault();
    
    const queryText = customText !== null ? customText : inputText;
    if (!queryText.trim()) return;

    setLoading(true);

    let updatedMessages = [...messages];

    if (isEdit && editIndex !== null) {
      // Edit past query: Delete subsequent chat history
      updatedMessages = messages.slice(0, editIndex);
      updatedMessages.push({ role: "user", content: queryText });
      setEditingMsgIndex(null);
    } else {
      updatedMessages.push({ role: "user", content: queryText });
      setInputText("");
    }

    setMessages(updatedMessages);

    // Extract history format for conversational memory API call
    const historyPayload = updatedMessages
      .slice(0, -1)
      .map((msg) => ({
        role: msg.role === "assistant" ? "assistant" : "user",
        content: msg.content,
      }));

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: queryText,
          target_file: selectedFile || null,
          history: historyPayload,
          check_relevance: checkRelevance,
        }),
      });

      const data = await res.json();

      if (res.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources || [],
            cache_hit: data.cache_hit,
            model_used: data.model_used,
            eval_metrics: data.eval_metrics,
            reformulated_query: data.reformulated_query,
          },
        ]);
        fetchStats();
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${data.detail || "Unable to get response from clinical models."}`,
            isError: true,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Failed to connect to the medical knowledge server. Please verify connections.",
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const triggerSampleQuery = (text) => {
    handleQuery(null, text);
  };

  const startEditing = (index, content) => {
    setEditingMsgIndex(index);
    setEditText(content);
  };

  const submitEdit = (e, index) => {
    e.preventDefault();
    if (!editText.trim()) return;
    handleQuery(null, editText, true, index);
  };

  const getGradeColor = (grade) => {
    switch (grade) {
      case "A": return "grade-a";
      case "B": return "grade-b";
      case "C": return "grade-c";
      default: return "grade-f";
    }
  };

  return (
    <div className="app-container">
      {/* ── LEFT PANEL: INGESTION & REGISTRY ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-logo">
            <h2>Healthcare RAG</h2>
          </div>
          <span className="badge-system">v4.0</span>
        </div>

        {/* File Ingest Section */}
        <section className="sidebar-section">
          <h3>Ingest Clinical Data</h3>
          <div className="upload-container">
            <label className="upload-box">
              <input
                type="file"
                accept=".pdf,.csv,.xlsx,.xls"
                onChange={handleFileUpload}
                disabled={uploading}
              />
              <div className="upload-content">
                <span className="upload-text" style={{ fontSize: uploading ? "12px" : "13px" }}>
                  {uploading ? uploadStage : "Upload Clinical File"}
                </span>
                <span className="upload-sub">PDF, CSV, Excel (max 50MB)</span>
              </div>
            </label>

            {uploading && (
              <div className="progress-bar-container">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
                <span className="progress-label">{uploadProgress}%</span>
              </div>
            )}

            {uploadResult && (
              <div className={`upload-status ${uploadResult.success ? "success" : "error"}`}>
                <div className="status-header">
                  <span>{uploadResult.success ? "Success" : "Error"}</span>
                  <button onClick={() => setUploadResult(null)}>Close</button>
                </div>
                <p className="status-msg">{uploadResult.message}</p>
                {uploadResult.success && (
                  <div className="status-metrics">
                    <span>Pages: {uploadResult.pages}</span>
                    <span>Time: {uploadResult.time}s</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* Document Registry Section */}
        <section className="sidebar-section files-section">
          <div className="section-header">
            <h3>Document Registry</h3>
            <button className="btn-icon-text" onClick={fetchFiles} title="Refresh Files">
              Refresh
            </button>
          </div>

          <div className="file-selector">
            <label className="filter-label">Search Targeting:</label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              className="styled-select"
            >
              <option value="">Global Search (All Files)</option>
              {files.map((f, i) => (
                <option key={i} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>

          <div className="file-list">
            {files.length === 0 ? (
              <p className="empty-text">No documents indexed in vector store.</p>
            ) : (
              files.map((file, idx) => {
                const ext = file.split(".").pop().toLowerCase();
                let fileLabel = "[DOC]";
                if (ext === "csv") fileLabel = "[CSV]";
                if (ext === "xlsx" || ext === "xls") fileLabel = "[XLS]";
                if (ext === "pdf") fileLabel = "[PDF]";

                return (
                  <div className="file-item" key={idx}>
                    <div className="file-info" title={file}>
                      <span className="file-icon-label">{fileLabel}</span>
                      <span className="file-name">{file}</span>
                    </div>
                    <button
                      className="delete-btn-text"
                      onClick={() => handleDeleteDocument(file)}
                      title="Delete document"
                    >
                      Delete
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </section>

        {/* MLOps Hard Reset Admin Control */}
        <section className="sidebar-section reset-section">
          <button className="btn-danger-outline" onClick={handleClearDatabase}>
            Clear Database Index
          </button>
        </section>
      </aside>

      {/* ── CENTER AREA: CHAT & CLINICAL INFERENCE ── */}
      <main className="main-content">
        <header className="main-header">
          <div className="header-info">
            <h1>Clinical Intelligence Console</h1>
            <span className="active-mode">
              {selectedFile ? `Targeted: ${selectedFile}` : "Global Search Mode"}
            </span>
          </div>

          {/* Relevance Checker Toggle */}
          <div className="header-controls">
            <label className="gatekeeper-toggle" title="Prevents out-of-scope queries using lightweight Llama evaluation">
              <input
                type="checkbox"
                checked={checkRelevance}
                onChange={(e) => setCheckRelevance(e.target.checked)}
              />
              <span className="slider-label">Relevance Gatekeeper</span>
            </label>
          </div>
        </header>

        {/* Chat Message Window */}
        <section className="chat-window">
          {messages.length === 0 ? (
            files.length === 0 ? (
              <div className="welcome-screen">
                <h2>Healthcare Assistant</h2>
                <p className="welcome-tagline">
                  Analyze clinical files, scanned reports, and standard guidelines with semantic medical indexing.
                </p>
                
                <div className="demo-alert-box">
                  <h4 className="guide-title" style={{ color: "#93c5fd", fontSize: "13px", fontWeight: "600", marginBottom: "8px" }}>
                    Onboarding Guide:
                  </h4>
                  <ul className="guide-steps" style={{ textAlign: "left", fontSize: "12px", color: "#94a3b8", paddingLeft: "20px", lineHeight: "1.6" }}>
                    <li>Upload patient histories, clinical notes, or WHO medical guidelines in the sidebar registry.</li>
                    <li>Toggle the Relevance Gatekeeper to intercept out-of-scope queries (optional).</li>
                    <li>Type your clinical inquiry or test diagnostic reasoning in the text input below.</li>
                  </ul>
                </div>
                <p className="waiting-status-text" style={{ fontStyle: "italic", fontSize: "12px", color: "var(--color-warning)", marginTop: "10px" }}>
                  Awaiting clinical document ingestion in the sidebar to enable querying.
                </p>
              </div>
            ) : (
              <div className="welcome-screen">
                <h2>Healthcare Assistant</h2>
                <p className="welcome-tagline">
                  Clinical records successfully ingested! You can now search across your documents, ask follow-up questions, or test diagnostic reasoning:
                </p>

                <div className="sample-queries" style={{ width: "100%", display: "flex", flexDirection: "column", gap: "8px", marginTop: "10px" }}>
                  <h4 className="queries-title" style={{ fontSize: "12px", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", textAlign: "left", marginBottom: "4px" }}>
                    Suggested Inquiries:
                  </h4>
                  <button onClick={() => triggerSampleQuery("What are the recommended treatments for malaria mentioned in the guidelines?")}>
                    Treatments for malaria (WHO Guidelines)
                  </button>
                  <button onClick={() => triggerSampleQuery("Summarize the chief complaint of patient Ram.")}>
                    Chief complaint for Ram (History Report)
                  </button>
                  <button onClick={() => triggerSampleQuery("Check the clinical findings in the medical record.")}>
                    Summarize medical record findings
                  </button>
                </div>
              </div>
            )
          ) : (
            <div className="messages-list">
              {messages.map((msg, index) => (
                <div key={index} className={`message-wrapper ${msg.role}`}>
                  <div className="avatar-label">
                    {msg.role === "user" ? "USER" : "CLINICAL AI"}
                  </div>

                  <div className="message-content">
                    {/* User Edit Mode */}
                    {msg.role === "user" ? (
                      editingMsgIndex === index ? (
                        <form onSubmit={(e) => submitEdit(e, index)} className="edit-form">
                          <textarea
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            className="edit-textarea"
                          />
                          <div className="edit-buttons">
                            <button type="submit" className="btn-save">Submit Edit</button>
                            <button type="button" className="btn-cancel" onClick={() => setEditingMsgIndex(null)}>Cancel</button>
                          </div>
                        </form>
                      ) : (
                        <div className="user-msg-container">
                          <p>{msg.content}</p>
                          <button
                            className="edit-msg-btn-text"
                            onClick={() => startEditing(index, msg.content)}
                            title="Edit prompt and rebuild chain"
                          >
                            Edit
                          </button>
                        </div>
                      )
                    ) : (
                      // AI response
                      <div className="assistant-msg-container">
                        {/* Reformulation Indicator */}
                        {msg.reformulated_query && (
                          <div className="reformulated-banner">
                            <span>Optimized Standalone Query: "{msg.reformulated_query}"</span>
                          </div>
                        )}

                        <p className="answer-text">{msg.content}</p>

                        {/* Citations / Sources */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="citations-block">
                            <h4>Retrieved Context Sources:</h4>
                            <div className="citations-list">
                              {msg.sources.map((src, i) => (
                                <span key={i} className="citation-badge">
                                  {src.source} {src.page ? `(Page ${src.page})` : ""} [{src.doc_type?.toUpperCase()}]
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* LLM-as-Judge Evaluator */}
                        {msg.eval_metrics && (
                          <div className="evaluator-wrapper">
                            <button
                              className="evaluator-toggle"
                              onClick={() => setExpandedEvalIndex(expandedEvalIndex === index ? null : index)}
                            >
                              <span>Quality Audit</span>
                              <span className={`grade-pill ${getGradeColor(msg.eval_metrics.eval_grade)}`}>
                                Grade {msg.eval_metrics.eval_grade}
                              </span>
                            </button>

                            {expandedEvalIndex === index && (
                              <div className="evaluator-details">
                                <div className="eval-metric-row">
                                  <span>Faithfulness (Groundedness):</span>
                                  <div className="metric-bar-bg">
                                    <div
                                      className="metric-bar-fill"
                                      style={{ width: `${msg.eval_metrics.faithfulness * 100}%`, backgroundColor: "#10b981" }}
                                    ></div>
                                  </div>
                                  <span className="metric-score">{Math.round(msg.eval_metrics.faithfulness * 100)}%</span>
                                </div>
                                <div className="eval-metric-row">
                                  <span>Answer Relevance:</span>
                                  <div className="metric-bar-bg">
                                    <div
                                      className="metric-bar-fill"
                                      style={{ width: `${msg.eval_metrics.answer_relevance * 100}%`, backgroundColor: "#3b82f6" }}
                                    ></div>
                                  </div>
                                  <span className="metric-score">{Math.round(msg.eval_metrics.answer_relevance * 100)}%</span>
                                </div>
                                <div className="eval-metric-row">
                                  <span>Context Precision (Retrieval Quality):</span>
                                  <div className="metric-bar-bg">
                                    <div
                                      className="metric-bar-fill"
                                      style={{ width: `${msg.eval_metrics.context_precision * 100}%`, backgroundColor: "#f59e0b" }}
                                    ></div>
                                  </div>
                                  <span className="metric-score">{Math.round(msg.eval_metrics.context_precision * 100)}%</span>
                                </div>
                                <div className="eval-meta">
                                  <span>Judge Model: {msg.eval_metrics.judge_model}</span>
                                  <span>Latency: {msg.eval_metrics.eval_latency_ms}ms</span>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Model tag */}
                        {msg.model_used && (
                          <div className="model-tag">
                            Inferred via {msg.model_used} {msg.cache_hit && "(Cached)"}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message-wrapper assistant loading">
                  <div className="avatar-label">CLINICAL AI</div>
                  <div className="message-content">
                    <div className="pulse-loader">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <p className="loading-text">Retrieving vector store and synthesizing diagnosis...</p>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </section>

        {/* Input Control Box */}
        <footer className="input-box">
          <form onSubmit={handleQuery} className="input-form">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask about patient reports, dosage guidelines, or medical comparisons..."
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleQuery(e);
                }
              }}
              disabled={loading}
            />
            <button type="submit" disabled={loading || !inputText.trim()} className="send-btn">
              Send
            </button>
          </form>
        </footer>
      </main>

      {/* ── RIGHT PANEL: MLOPS OBSERVABILITY ── */}
      <aside className="observability-panel">
        <div className="obs-header">
          <h3>Observability Console</h3>
          <span className="pulse-badge-text">Telemetry Active</span>
        </div>

        {/* Live Index parameters */}
        <section className="obs-section">
          <h4>Vector Store Stats</h4>
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-num">{stats.upload_count}</span>
              <span className="stat-lbl">Indexed Files</span>
            </div>
            <div className="stat-card">
              <span className="stat-num">{stats.index_size_mb} MB</span>
              <span className="stat-lbl">FAISS Size</span>
            </div>
          </div>
        </section>

        {/* Prompt cache panel */}
        <section className="obs-section">
          <h4>Prompt Cache (SHA-256)</h4>
          <div className="cache-stats">
            <div className="cache-row">
              <span>Cached Queries:</span>
              <span className="bold">{stats.prompt_cache.cached_queries}</span>
            </div>
            <div className="cache-row">
              <span>Cache Hits:</span>
              <span className="bold text-emerald">{stats.prompt_cache.total_cache_hits}</span>
            </div>
            <div className="cache-row">
              <span>Hit Rate:</span>
              <span className="bold text-blue">{stats.token_budget.cache_hit_rate_pct}%</span>
            </div>
          </div>
        </section>

        {/* Token Budget Tracker & Estimator */}
        <section className="obs-section">
          <h4>Token Budget & Costs</h4>
          <div className="token-budget-details">
            <div className="cost-showcase">
              <span className="cost-title">Session Estimated Cost</span>
              <span className="cost-num">${stats.token_budget.estimated_cost_usd.toFixed(6)}</span>
              <span className="cost-currency">USD</span>
            </div>

            <div className="token-metrics">
              <div className="metric-pair">
                <span>Input Tokens:</span>
                <span>{stats.token_budget.total_input_tokens.toLocaleString()}</span>
              </div>
              <div className="metric-pair">
                <span>Output Tokens:</span>
                <span>{stats.token_budget.total_output_tokens.toLocaleString()}</span>
              </div>
            </div>

            {/* Model Distribution Router progress */}
            <div className="model-distribution">
              <span className="lbl">Model Router Allocations:</span>
              <div className="router-bar">
                <div
                  className="bar-8b"
                  style={{
                    width: `${
                      stats.token_budget.api_calls_made > 0
                        ? (stats.token_budget.model_8b_calls / stats.token_budget.api_calls_made) * 100
                        : 50
                    }%`,
                  }}
                  title={`Llama-3 8B: ${stats.token_budget.model_8b_calls} calls`}
                ></div>
                <div
                  className="bar-70b"
                  style={{
                    width: `${
                      stats.token_budget.api_calls_made > 0
                        ? (stats.token_budget.model_70b_calls / stats.token_budget.api_calls_made) * 100
                        : 50
                    }%`,
                  }}
                  title={`Llama-3 70B: ${stats.token_budget.model_70b_calls} calls`}
                ></div>
              </div>
              <div className="router-legend">
                <span className="legend-8b">8B ({stats.token_budget.model_8b_calls})</span>
                <span className="legend-70b">70B ({stats.token_budget.model_70b_calls})</span>
              </div>
            </div>
          </div>
        </section>

        {/* System logs info */}
        <footer className="obs-footer">
          <span className="logs-lbl">Telemetry logs:</span>
          <span className="logs-path">{stats.log_file}</span>
        </footer>
      </aside>
    </div>
  );
}

export default App;
