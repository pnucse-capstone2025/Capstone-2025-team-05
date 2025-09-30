// SignupPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/LoginPage.css';
import logo from '../assets/logo.png';

function SignupPage() {
  const [step, setStep] = useState(1);      // 진행 단계 (1: 이메일 입력, 2: 코드 입력, 3: 가입정보 입력, 4: 완료)
  const navigate = useNavigate();

  // 입력 값 상태
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [nickname, setNickname] = useState(''); 

  // 1. 인증 코드 이메일 발송
  const handleSendVerification = async () => {
    if (!email) {
      alert("Please enter your email.");
      return;
    }

    try {
      await axios.post("http://localhost:5001/auth/send-code", { email });
      alert("Verification code sent to your email.");
      setStep(2);
    } catch (error) {
      console.error("❌ Send code failed:", error);
      alert(error.response?.data?.error || "Failed to send verification code.");
    }
  };

  // 2. 인증 코드 확인
  const handleVerifyCode = async () => {
    if (!code) {
      alert("Please enter the verification code.");
      return;
    }

    try {
      await axios.post("http://localhost:5001/auth/verify-code", { email, code });
      alert("Email verified successfully!");
      setStep(3);
    } catch (error) {
      console.error("❌ Verify code failed:", error);
      alert(error.response?.data?.error || "Failed to verify code.");
    }
  };

  // 3. 회원가입
  const handleSignup = async () => {
    if (!nickname) {
      alert("Please enter your nickname.");
      return;
    }

    if (password.length < 8) {
      alert("Password must be at least 8 characters.");
      return;
    }

    if (password !== confirmPwd) {
      alert("Passwords do not match.");
      return;
    }

    try {
      await axios.post("http://localhost:5001/auth/signup", {
        email,
        password,
        nickname, 
      });
      alert("Signup completed successfully!");
      setStep(4);
    } catch (error) {
      console.error("❌ Signup failed:", error);
      alert(error.response?.data?.error || "Signup failed.");
    }
  };

  return (
    <div className="login-page">
      <img src={logo} alt="Logo" className="login-logo" />

      <div className="login-form">
        {/* 1단계: 이메일 입력 */}
        {step === 1 && (
          <>
            <input
              type="email"
              placeholder="Enter your Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button className="full-width-button" onClick={handleSendVerification}>
              Send Code
            </button>
          </>
        )}
        {/* 2단계: 코드 입력 */}
        {step === 2 && (
          <>
            <input
              type="text"
              placeholder="Enter Code"
              value={code}
              onChange={(e) => setCode(e.target.value)}
            />
            <button className="full-width-button" onClick={handleVerifyCode}>
              Verify Code
            </button>
          </>
        )}
        {/* 3단계: 닉네임 + 비밀번호 설정 */}
        {step === 3 && (
          <>
            <input
              type="text"
              placeholder="Nickname"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
            />
            <input
              type="password"
              placeholder="New Password (min 8 characters)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <input
              type="password"
              placeholder="Confirm Password"
              value={confirmPwd}
              onChange={(e) => setConfirmPwd(e.target.value)}
            />
            <button className="full-width-button" onClick={handleSignup}>
              Create Account
            </button>
          </>
        )}
        {/* 4단계: 완료 후 로그인 이동 */}
        {step === 4 && (
          <>
            <p style={{ fontSize: '1.2rem', textAlign: 'center' }}>
              
            </p>
            <button className="full-width-button" onClick={() => navigate('/login')}>
              Go to Login
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default SignupPage;
