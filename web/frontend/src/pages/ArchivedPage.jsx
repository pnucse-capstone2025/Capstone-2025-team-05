// ArchivedPage.jsx
import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/ArchivedPage.css";

function ArchivedPage() {
  const [predictions, setPredictions] = useState([]);   // 아카이브된 예측 리스트
  const [loading, setLoading] = useState(true);         // 로딩 상태 관리
  const navigate = useNavigate();

  const nickname = localStorage.getItem("nickname");    // 닉네임 불러오기

  // 로그아웃 시 토큰과 닉네임 제거 후 로그인 페이지로 이동
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("nickname");
    navigate("/login");
  };


  // 마운트 시 아카이브된 예측 불러오기
  useEffect(() => {
    const fetchArchived = async () => {
      try {
        const res = await fetch("http://localhost:5001/predictions/archived", {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        });
        if (!res.ok) throw new Error("Failed to fetch archived predictions");
        const data = await res.json();
        setPredictions(data);
      } catch (err) {
        console.error("❌ Error fetching archived:", err);
        alert("아카이빙 불러오기 실패");
      } finally {
        setLoading(false);
      }
    };
    fetchArchived();
  }, []);

  if (loading) return <div>⏳ 불러오는 중...</div>;

  return (
    <div className="archived-page">
      {/* 상단바: Home 버튼 + 닉네임/로그아웃 */}
      <div className="archived-top-bar">
        <div className="archived-left">
          <button
            className="archived-home-btn"
            onClick={() => navigate("/input")}
          >
            🏠
          </button>
        </div>

        {nickname && (
          <div className="archived-nickname-display">
            {nickname}
            <button
              className="archived-logout-btn"
              onClick={handleLogout}
            >
              Logout
            </button>
          </div>
        )}
      </div>

      <h1>Archived Predictions</h1>
      {predictions.length === 0 ? (
        <p>저장된 예측이 없습니다.</p>
      ) : (
        <div className="archived-list">
          {predictions.map((p) => (
            <div
              key={p.id}
              className="archived-card"
              onClick={() =>
                navigate("/prediction", {
                  state: { predictionId: p.id, protname: p.protname },
                })
              }
            >
              <h3>{p.protname}</h3>
              <p>
                <b>Affinity:</b>{" "}
                {p.affinity ? p.affinity.toFixed(2) : "N/A"}
              </p>
              <p>
                <b>Date:</b> {p.created_at}
              </p>

              {/* 예측 스크린샷 표시 */}
              {p.screenshot_url && (
                <img
                  src={p.screenshot_url}
                  alt="Prediction Screenshot"
                  className="ngl-thumb-container"
                />
              )}

            </div>
          ))}
        </div>
      )}
    </div>
  );
}


export default ArchivedPage;
