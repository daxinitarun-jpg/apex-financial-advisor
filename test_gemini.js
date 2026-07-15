const { GoogleGenerativeAI } = require("@google/generative-ai");
try {
  const genAI = new GoogleGenerativeAI("invalid_key_123");
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash", systemInstruction: "hello" });
  console.log("No error thrown during init.");
} catch(e) {
  console.log("Error thrown:", e.message);
}
