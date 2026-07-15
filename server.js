require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' })); // Allow large payloads for PDF text

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const SYSTEM_PROMPT = `You are Apex Bro, an elite Wall Street hedge fund manager and world-class financial advisor. You provide absolutely flawless, mathematically perfect financial analysis and highly complex strategic advice. 
When the user provides a PDF or financial data, read it perfectly. Calculate deep insights.
Format your response beautifully using HTML tags like <h3>, <table>, <tr>, <th>, <td>, and highlight boxes using classes like 'recommendation-box', 'success-box', 'warning-box', or 'danger-box'. Only use inline CSS or HTML. NEVER use Markdown (no \`\`\`html blocks). Just output raw HTML that can be inserted directly into a <div>. Focus on advanced profitability, ratios, margins, risk assessments, and elite actionable strategy. Never give investment advice.`;

app.post('/api/chat', async (req, res) => {
    try {
        const { message, history, attachments } = req.body;
        
        let historyContext = "";
        if (history && history.length > 0) {
            for(const msg of history.slice(-4)) {
                if(msg.role === 'user') historyContext += `User: ${msg.content}\n`;
                else historyContext += `Apex Bro: ${msg.content}\n`;
            }
        }
        
        let finalPrompt = message;
        if(historyContext.length > 0) {
            finalPrompt = `[Previous Conversation Context]:\n${historyContext}\n\n[New User Message]:\n${message}`;
        }

        const parts = [{ text: finalPrompt }];
        if (attachments && attachments.length > 0) {
            for(const att of attachments) {
                if(att.isImage && att.data) parts.push({ inlineData: { data: att.data.split(',')[1], mimeType: att.type } });
                else if (att.extractedText) parts.push({ text: `\n\n--- Document: ${att.name} ---\n${att.extractedText}\n--- End Document ---` });
            }
        }

        // Try multiple models in order until one works
        const modelsToTry = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ];

        let lastError = null;
        for (const modelName of modelsToTry) {
            try {
                console.log(`Trying model: ${modelName}...`);
                const config = modelName === "gemini-pro" ? {} : { systemInstruction: SYSTEM_PROMPT };
                const model = genAI.getGenerativeModel({ model: modelName, ...config });
                
                let finalParts = [...parts];
                if (modelName === "gemini-pro") {
                    // gemini-pro doesn't support systemInstruction, prepend it
                    finalParts = [{ text: "SYSTEM INSTRUCTION: " + SYSTEM_PROMPT + "\n\n" + finalPrompt }];
                }

                const result = await model.generateContent(finalParts);
                let text = result.response.text();
                
                // Clean up any markdown blocks
                text = text.replace(/^```html\n?/, '').replace(/^```\n?/, '').replace(/```$/, '');
                
                console.log(`Success with model: ${modelName}`);
                return res.json({ response: text });
            } catch (modelError) {
                console.warn(`Model ${modelName} failed:`, modelError.message);
                lastError = modelError;
            }
        }
        
        // If all models failed
        throw lastError;

    } catch (error) {
        console.error("Error in /api/chat:", error);
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
