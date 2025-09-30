// PredictionViewPage.jsx
import React, { useEffect, useState, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom"; 
import "../styles/PredictionViewPage.css";
import "../styles/Spinner.css";

function PredictionViewPage() {
  const location = useLocation();
  const navigate = useNavigate(); 
  const { predictionId, protname } = location.state || {};
  

  const stageRef = useRef(null);
  const containerRef = useRef(null);
  const selectedResnosRef = useRef(new Set());
  const flickerIntervalRef = useRef(null);
  
  const [archived, setArchived] = useState(false);
  const [loading, setLoading] = useState(true);
  const [sequence, setSequence] = useState("");
  const [bindingSites, setBindingSites] = useState([]);
  const [affinity, setAffinity] = useState(null);
  const [pdbUrl, setPdbUrl] = useState(null);

  const affinityPercent = affinity
    ? Math.min(Math.round((affinity / 10) * 100), 100)
    : 0;

  const nickname = localStorage.getItem("nickname");

  // 로그아웃
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("nickname");
    navigate("/login");
  };

  // SSE 구독 (예측 스트리밍)
  useEffect(() => {
    if (!predictionId) return;

    const token = localStorage.getItem("token");
    const evtSource = new EventSource(
      `http://localhost:5001/predict/stream/${predictionId}?token=${token}`
    );

    evtSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.stage === "quick") {
        // 1차 결과: 시퀀스, 결합부위, 친화도 표시
        setSequence(data.sequence || "");
        setBindingSites(data.binding_sites || []);
        setAffinity(data.affinity);
        setLoading(false); 
      } else if (data.stage === "final") {
        // 최종 결과: 구조 URL 수신
        console.log("📡 최종 이벤트 수신:", data);
        if (data.pdb_url) {
          console.log("✅ 받은 pdb_url:", data.pdb_url);
          setPdbUrl(data.pdb_url); 
        } else {
          console.warn("⚠️ pdb_url 없음");
        }
        evtSource.close();
      }
    };

    evtSource.onerror = (err) => {
      console.error("❌ SSE 오류:", err);
      evtSource.close();
      if (err?.status === 401) {
        alert("세션이 만료되었습니다. 다시 로그인하세요.");
        handleLogout();
      } else {
        alert("서버 오류가 발생했습니다. 다시 시도해주세요.");
        navigate("/input");
      }
    };

    return () => evtSource.close();
  }, [predictionId, navigate]);

  // NGL 잔기 깜빡임 효과
  const startFlicker = (component, selection) => {
    let isWhite = true;
    if (flickerIntervalRef.current) {
      clearInterval(flickerIntervalRef.current);
    }
    flickerIntervalRef.current = setInterval(() => {
      try {
        if (!component || !selection) return;

        const prev = component.reprList.find((r) => r.name === "selected");
        if (prev) component.removeRepresentation(prev);

        component.addRepresentation("ball+stick", {
          sele: selection,
          name: "selected",
          color: isWhite ? "white" : "red",
          opacity: 1.0,
          side: "double",
          depthTest: false,
          radiusScale: 1.5,
        });

        isWhite = !isWhite;
      } catch (err) {
        console.warn("⚠️ 깜빡임 중 에러 발생:", err);
      }
    }, 500);
  };

  // NGL 뷰어 초기화
  const initNGL = async (selected = []) => {
    if (!pdbUrl) return;

    if (stageRef.current) {
      stageRef.current.dispose();
      containerRef.current.innerHTML = "";
    }

    const stage = new window.NGL.Stage(containerRef.current, {
      backgroundColor: "black",
    });
    stageRef.current = stage;

    const comp = await stage.loadFile(pdbUrl, { defaultRepresentation: false });
    const structure = comp.structure;
    if (!structure) return;

    const validResnos = new Set();
    structure.eachResidue((rp) => validResnos.add(rp.resno));

    const filteredBinding = bindingSites.filter((r) => validResnos.has(r));
    const bindingSelection = filteredBinding.map((r) => `resi ${r}`).join(" OR ");

    const filteredSelected = selected.filter((r) => validResnos.has(r));
    const selectedSelection = filteredSelected.map((r) => `resi ${r}`).join(" OR ");

    // 전체 구조
    comp.addRepresentation("ball+stick", {
      color: "white",
      radiusScale: 1.5,
    });

    // 결합 부위
    if (bindingSelection) {
      comp.addRepresentation("ball+stick", {
        sele: bindingSelection,
        name: "binding",
        color: "skyblue",
        radiusScale: 1.5,
      });
    }

    // 선택된 부위
    if (selectedSelection) {
      comp.addRepresentation("ball+stick", {
        sele: selectedSelection,
        name: "selected",
        color: "red",
        opacity: 1.0,
        side: "double",
        depthTest: false,
        radiusScale: 1.5,
      });
      startFlicker(comp, selectedSelection);
    } else {
      if (flickerIntervalRef.current) {
        clearInterval(flickerIntervalRef.current);
        flickerIntervalRef.current = null;
      }
    }

    stage.autoView();
  };

  // 구조 로딩 시 초기화 실행
  useEffect(() => {
    if (pdbUrl) {
      initNGL(Array.from(selectedResnosRef.current));
    }
  }, [pdbUrl, bindingSites]);

  // 잔기 클릭 처리
  const handleResidueClick = (resIndex) => {
    if (!bindingSites.includes(resIndex)) return;

    const selected = selectedResnosRef.current;
    const isSelected = selected.has(resIndex);

    isSelected ? selected.delete(resIndex) : selected.add(resIndex);

    const elem = document.getElementById(`res-${resIndex}`);
    if (elem) {
      elem.classList.toggle("selected-site", !isSelected);
    }

    const updated = Array.from(selected);
    initNGL(updated);

    console.log("✨ 선택된 레지듀:", updated);
  };

  return (
    <div className="prediction-page">
      {/* 로딩 상태 */}
      {loading ? (
        <div className="spinner-wrapper">
          <h1 className="spinner-title">{protname || "Prediction"}</h1>
          <div className="spinner"></div>
          <div className="spinner-text">🧬 예측 중...</div>
        </div>
      ) : (
        <>
          {/* 상단바 */}
          <div className="prediction-top-bar">
            <div className="prediction-left-buttons">
              <button className="prediction-home-btn" onClick={() => navigate("/input")}>
                🏠
              </button>
              <button className="prediction-archive-btn" onClick={() => navigate("/archived")}>
                Archive
              </button>
            </div>

            {nickname && (
              <div className="prediction-nickname">
                {nickname}
                <button className="prediction-logout-btn" onClick={handleLogout}>
                  logout
                </button>
              </div>
            )}
          </div>

          {/* 단백질 이름 */}
          <h1>{protname || "Protein Prediction"}</h1>

          {/* 구조 뷰어 */}
          <div id="ngl-container" ref={containerRef}>
            {!pdbUrl && (
              <div className="ngl-loading">
                <div className="spinner2"></div>
                <p>Predicting structure...</p>
              </div>
            )}
          </div>

          {/* 시퀀스 표시 */}
          <div className="sequence-container">
            {sequence.split("").map((residue, idx) => {
              const resIndex = idx + 1;
              const isBindingSite = bindingSites.includes(resIndex);
              return (
                <span
                  key={idx}
                  id={`res-${resIndex}`}
                  className={`residue ${isBindingSite ? "binding-site" : ""}`}
                  onClick={() => handleResidueClick(resIndex)}
                >
                  {residue}
                </span>
              );
            })}
          </div>

          {/* 친화도 바 */}
          <div className="affinity-score-container">
            <div className="affinity-score-label">Binding Affinity</div>
            <div className="affinity-bar-wrapper">
              <div
                className="affinity-bar-fill"
                style={{ width: `${affinityPercent}%` }}
              >
                <span className="affinity-score-value">
                  {affinity?.toFixed(2)}
                </span>
              </div>
            </div>
          </div>

          {/* 저장 버튼 */}
          <button
            className="prediction-save-btn"
            disabled={archived}
            onClick={async () => {
              try {
                if (!stageRef.current) {
                  alert("❌ 구조가 로드되지 않아 캡처할 수 없습니다.");
                  return;
                }

                // NGL 캡처
                const blob = await stageRef.current.makeImage({
                  factor: 2,        // 해상도 배율 (2배)
                  antialias: true,
                  trim: false,
                  transparent: false,
                });

                // FormData 생성
                const formData = new FormData();
                formData.append("file", blob, `screenshot_${predictionId}.png`);

                // 업로드 요청
                const uploadRes = await fetch(
                  `http://localhost:5001/upload/screenshot/${predictionId}`,
                  {
                    method: "POST",
                    headers: {
                      Authorization: `Bearer ${localStorage.getItem("token")}`,
                    },
                    body: formData,
                  }
                );

                if (!uploadRes.ok) {
                  throw new Error("스크린샷 업로드 실패");
                }
                const uploadData = await uploadRes.json();
                console.log("✅ Screenshot uploaded:", uploadData.url);

                // Archive API 호출
                const res = await fetch(`http://localhost:5001/predictions/${predictionId}/archive`, {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${localStorage.getItem("token")}`,
                  },
                });
                if (res.ok) {
                  setArchived(true);
                  alert("✅ Saved to your archive!");
                } else {
                  const err = await res.json().catch(() => ({}));
                  alert(`❌ Save failed: ${err.error || res.status}`);
                }
              } catch (e) {
                console.error("Archive error:", e);
                alert("❌ Network error while saving.");
              }
            }}
          >
            {archived ? "Saved" : "Save"}
          </button>


        </>
      )}
    </div>
  );
}

export default PredictionViewPage;
