// InputPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";   
import "../styles/InputPage.css";
import logo from "../assets/logo.png";

function InputPage() {
  const [proteinSeq, setProteinSeq] = useState("");          // 단백질 시퀀스 입력
  const [ligandSmiles, setLigandSmiles] = useState("");      // 리간드 SMILES 입력
  const [protname, setProtname] = useState("");              // 단백질 이름 입력
  const navigate = useNavigate();

  const nickname = localStorage.getItem("nickname");         // 사용자 닉네임 불러오기

  // 로그아웃 처리
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("nickname");
    navigate("/login");
  };

  // 예측 요청
  const handleSubmit = async () => {
    if (!protname || !proteinSeq || !ligandSmiles) {
      alert("⚠️ Please enter protein name, sequence, and ligand SMILES.");
      return;
    }

    console.log("🚀 Protein Sequence:", proteinSeq);
    console.log("💊 Ligand SMILES:", ligandSmiles);

    try {
      const token = localStorage.getItem("token");

      // 서버에 예측 시작 요청 → prediction_id 반환
      const response = await axios.post(
        "http://localhost:5001/predict/start",
        {
          protname,
          sequence: proteinSeq,
          smiles: ligandSmiles,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // ✅ 토큰 추가
          },
        }
      );

      const predictionId = response.data.prediction_id;
      console.log("✅ Prediction started, ID:", predictionId);

      // PredictionViewPage로 이동
      navigate("/prediction", {
        state: {
          predictionId,
          protname,
        },
      });
    } catch (error) {
      console.error("❌ Prediction start failed:", error);
      alert("Prediction request failed.");
    }
  };

  return (
    <div className="input-page">
      {/* 상단바: Archive 버튼 + 닉네임/로그아웃 */}
      <div className="input-top-bar">
        <div className="input-left-buttons">
          <button
            className="input-archive-btn"
            onClick={() => navigate("/archived")}
          >
            Archive
          </button>
        </div>

        {nickname && (
          <div className="input-nickname">
            {nickname}
            <button className="input-logout-btn" onClick={handleLogout}>
              Logout
            </button>
          </div>
        )}
      </div>


      {/* 로고 */}
      <img src={logo} alt="Planet-X Logo" className="input-logo" />

      {/* 입력 폼 */}
      <div className="form-wrapper">
        <label htmlFor="protname">🧾 Protein Name</label>
        <input
          id="protname"
          type="text"
          value={protname}
          onChange={(e) => setProtname(e.target.value)}
        />

        <label htmlFor="protein-seq">🧬 Protein Sequence (FASTA)</label>
        <textarea
          id="protein-seq"
          rows={8}
          placeholder="e.g., MQDRVKRPMNAFIVWSRDQRRKMALEN..."
          value={proteinSeq}
          onChange={(e) => setProteinSeq(e.target.value)}
        />

        <label htmlFor="ligand-smiles">💊 Ligand SMILES</label>
        <input
          id="ligand-smiles"
          type="text"
          placeholder="e.g., C1=CC=CC=C1"
          value={ligandSmiles}
          onChange={(e) => setLigandSmiles(e.target.value)}
        />

        <button className="submit-button" onClick={handleSubmit}>
          RUN PREDICTION
        </button>
      </div>
    </div>
  );
}

export default InputPage;
