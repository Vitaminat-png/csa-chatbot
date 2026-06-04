window.AVATAR_POC_CONFIG = {
  chatbotApiBase: "https://csa-chatbot.onrender.com",
  avatarBackendBase: "http://127.0.0.1:8000",
  provider: "simli",
  defaultFaceId: "csa-anna",
  defaultVoiceId: "coral",
  reconnectOnAvatarChange: true,
  simliSession: {
    maxSessionLength: 900,
    maxIdleTime: 120,
    handleSilence: true,
    model: "fasttalk"
  },
  tts: {
    streamChunkBytes: 6000,
    sentenceFlushChars: 120,
    minFlushChars: 48,
    instructions: "Parla in italiano con voce femminile naturale, tono professionale, dizione chiara e ritmo medio."
  },
  faces: [
    {
      id: "csa-anna",
      label: "Anna",
      simliFaceId: "REPLACE_WITH_SIMLI_FACE_ID_1",
      previewImage: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=640&q=80",
      description: "Fotorealistica · volto 1"
    },
    {
      id: "csa-sofia",
      label: "Sofia",
      simliFaceId: "REPLACE_WITH_SIMLI_FACE_ID_2",
      previewImage: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=640&q=80",
      description: "Fotorealistica · volto 2"
    }
  ],
  voices: [
    {
      id: "coral",
      label: "Coral",
      language: "it",
      description: "OpenAI TTS · femminile naturale"
    },
    {
      id: "shimmer",
      label: "Shimmer",
      language: "it",
      description: "OpenAI TTS · femminile brillante"
    }
  ],
  samplePrompts: [
    "Quali valvole CSA consigli per alte temperature?",
    "Spiegami la differenza tra valvole wafer e lug.",
    "Quali certificazioni avete per il settore industriale?"
  ]
};