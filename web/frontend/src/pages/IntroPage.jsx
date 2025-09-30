// IntroPage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/IntroPage.css';  
import logo from '../assets/logo.png';

function IntroPage() {
  const navigate = useNavigate();

  return (
    // 로고 화면 → 클릭 시 로그인 페이지로 이동
    <div className="intro-container" onClick={() => navigate('/login')}>
      <img src={logo} alt="Planet X Logo" className="intro-logo" />
      
    </div>
  );
}

export default IntroPage;
