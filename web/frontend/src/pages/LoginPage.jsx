// LoginPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios'; 
import '../styles/LoginPage.css';
import logo from '../assets/logo.png';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  // 로그인 요청
  const handleLogin = async () => {
    try {
      const response = await axios.post("http://localhost:5001/auth/login", {
        email,
        password,
      });

      const { token, nickname } = response.data;

      // 토큰/닉네임 저장 → 이후 인증 요청에 사용
      localStorage.setItem("token", token);
      localStorage.setItem("nickname", nickname);

      alert(`Welcome, ${nickname}!`);
      navigate("/input"); // 로그인 성공 시 입력 페이지로 이동
    } catch (error) {
      console.error("❌ Login failed:", error);
      alert(error.response?.data?.error || "Login failed. Please try again.");
    }
  };

  // 회원가입 페이지로 이동
  const handleSignup = () => {
    navigate('/signup'); 
  };

  return (
    <div className="login-page">
      {/* 로고 */}      
      <img src={logo} alt="Logo" className="login-logo" />

      {/* 로그인 폼 */} 
      <div className="login-form">
        <input
          type="email"
          placeholder="이메일"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {/* 버튼 영역 */}
        <div className="login-buttons">
          <button className="login-btn" onClick={handleLogin}>로그인</button>
          <button className="signup-btn" onClick={handleSignup}>회원가입</button>
        </div>

      </div>
    </div>
  );
}

export default LoginPage;
