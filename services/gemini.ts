
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export async function getTutorResponse(userMessage: string, history: {role: 'user' | 'assistant', content: string}[]) {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: [
        { 
          role: 'user', 
          parts: [{ text: `You are an expert econometrics tutor. You are teaching a first-year student about Decision Trees using an example of predicting "Job Found" based on "Prior Income" and "Education". 
          
          Context of the specific model being discussed:
          - Root Split: Income < 20k
          - If Income >= 20k: Split on Education < 16. If yes, predict No Job. If no, predict Job Found.
          - If Income < 20k: Split on Education < 12. If yes, predict No Job. If no, predict Job Found.
          
          Guidelines:
          1. Use econometric terminology (non-linearity, heterogeneity, interactions, conditional expectation).
          2. Explain how this differs from a simple linear regression (OLS).
          3. Keep it accessible to a beginner.
          4. If they ask about the image, reference the logic above.
          
          History: ${JSON.stringify(history)}
          
          Question: ${userMessage}` }] 
        }
      ],
      config: {
        temperature: 0.7,
        topP: 0.8,
      }
    });

    return response.text;
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "I'm sorry, I'm having trouble connecting to my econometric brain right now. Let's try again in a moment!";
  }
}
