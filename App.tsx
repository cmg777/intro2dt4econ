import React, { useState, useMemo } from 'react';
import ScatterPlot from './components/ScatterPlot';
import DecisionTreeVisualizer from './components/DecisionTreeVisualizer';
import { INITIAL_DATA } from './constants';

const PYTHON_CODE = `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==============================================================================
# PROFESSOR'S NOTE: THE GOAL
# In econometrics, we often care about 'inference' (understanding relationships).
# In machine learning, we care about 'prediction' (guessing outcomes for new data).
# This workflow is designed to ensure our predictions are honest and robust.
# ==============================================================================

# ==========================================
# STEP 1: DATA GENERATION (The Sample)
# ==========================================
# In a real study, you would load this from a CSV file (e.g., pd.read_csv).
# Here, we define 24 individuals representing our labor market sample.
data = [
    # -- High Income (>= $20k), High Education (>= 16y) -> Job Found --
    {'education': 18, 'income': 80, 'found_job': 1},
    {'education': 17, 'income': 70, 'found_job': 1},
    {'education': 19, 'income': 40, 'found_job': 1},
    {'education': 16.5, 'income': 60, 'found_job': 1},
    
    # -- High Income (>= $20k), Low Education (< 16y) -> No Job --
    {'education': 5, 'income': 85, 'found_job': 0},
    {'education': 8, 'income': 75, 'found_job': 0},
    {'education': 4, 'income': 65, 'found_job': 0},
    {'education': 12, 'income': 80, 'found_job': 0},
    {'education': 10, 'income': 55, 'found_job': 0},
    {'education': 14, 'income': 70, 'found_job': 0},
    {'education': 2, 'income': 45, 'found_job': 0},
    {'education': 7, 'income': 30, 'found_job': 0},

    # -- Low Income (< $20k), High Education (>= 12y) -> Job Found --
    {'education': 14, 'income': 15, 'found_job': 1},
    {'education': 16, 'income': 10, 'found_job': 1},
    {'education': 18, 'income': 5, 'found_job': 1},
    {'education': 12, 'income': 8, 'found_job': 1},
    {'education': 15, 'income': 12, 'found_job': 1},
    {'education': 19, 'income': 2, 'found_job': 1},
    {'education': 13, 'income': 18, 'found_job': 1},

    # -- Low Income (< $20k), Low Education (< 12y) -> No Job --
    {'education': 4, 'income': 15, 'found_job': 0},
    {'education': 8, 'income': 10, 'found_job': 0},
    {'education': 2, 'income': 5, 'found_job': 0},
    {'education': 6, 'income': 12, 'found_job': 0},
    {'education': 1, 'income': 18, 'found_job': 0},
]

df = pd.DataFrame(data)

# Features (X) = The Independent Variables (Covariates)
# Target (y)   = The Dependent Variable (Outcome)
X = df[['income', 'education']] 
y = df['found_job']

# ==========================================
# STEP 2: IDENTIFICATION (Train-Test Split)
# ==========================================
# PROFESSOR'S NOTE: This is the most critical step.
# We separate the data into two "universes."
# 1. Training Set (70%): The textbook the model studies to learn rules.
# 2. Test Set (30%): The final exam. The model NEVER sees this data during training.
#
# 'stratify=y' ensures that if 50% of people have jobs in reality, 
# both our Training and Test sets reflect that 50/50 split. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples (The Study Guide): {len(X_train)}")
print(f"Testing samples (The Final Exam):   {len(X_test)}")

# ==========================================
# STEP 3: ESTIMATION (Model Fitting)
# ==========================================
# We initialize an empty Decision Tree.
# 'max_depth=3' is a hyperparameter. It prevents the tree from becoming too complex
# (memorizing the data) by limiting it to only 3 levels of "questions."
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# We fit the model ONLY on X_train. 
# It is completely blind to X_test at this stage.
clf.fit(X_train, y_train)

# ==========================================
# STEP 4: DIAGNOSTICS (Evaluation)
# ==========================================
# Now we force the model to take the "Final Exam."
# We feed it the Test Features (X_test) and ask for predictions.
y_pred = clf.predict(X_test)

print("\\n--- Model Performance on Test Data ---")
# Accuracy: Simple percentage of correct guesses.
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Classification Report:
# Precision: When it predicts "Job", how often is it right?
# Recall: Out of everyone who *actually* found a job, how many did it catch?
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Job', 'Found Job']))

# ==========================================
# STEP 5: VISUALIZATION (Logic Inspection)
# ==========================================
# This plot shows us the "rules" the model invented.
# Example: "Is Income <= 19.0? If yes, check Education..."
plt.figure(figsize=(12, 6))
plot_tree(clf, 
          feature_names=['Income', 'Education'], 
          class_names=['No Job', 'Found Job'], 
          filled=True, 
          rounded=True)
plt.title("Decision Tree Logic (Learned from Training Data Only)")
plt.show()

# ==========================================
# STEP 6: VISUALIZATION (The Decision Boundary)
# ==========================================
# PROFESSOR'S NOTE: Pay attention to the shapes!
# CIRCLES = Training Data (What the model saw)
# STARS   = Testing Data (What the model is guessing on)
# If a STAR is in the wrong color region, that is a Prediction Error.

plt.figure(figsize=(10, 6))

# A. Create the background mesh (The Model's Worldview)
x_min, x_max = 0, 100 # Income range
y_min, y_max = 0, 20  # Education range
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.1))

# Predict for every point in the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot contours (The Decision Regions)
plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlGn')

# Prepare dataframes for easier plotting
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# B. Plot TRAINING data (Circles)
plt.scatter(train_df[train_df['found_job']==1]['income'], train_df[train_df['found_job']==1]['education'], 
            c='green', marker='o', s=100, edgecolors='k', label='Train: Job (Seen)')
plt.scatter(train_df[train_df['found_job']==0]['income'], train_df[train_df['found_job']==0]['education'], 
            c='red', marker='o', s=100, edgecolors='k', label='Train: No Job (Seen)')

# C. Plot TESTING data (Stars)
plt.scatter(test_df[test_df['found_job']==1]['income'], test_df[test_df['found_job']==1]['education'], 
            c='green', marker='*', s=200, edgecolors='k', label='Test: Job (Unseen)')
plt.scatter(test_df[test_df['found_job']==0]['income'], test_df[test_df['found_job']==0]['education'], 
            c='red', marker='*', s=200, edgecolors='k', label='Test: No Job (Unseen)')

plt.ylabel('Education (Years)')
plt.xlabel('Prior Income (k$)')
plt.title('Decision Boundary: Visualizing Generalization')
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.show()`;

const App: React.FC = () => {
  // Model Parameters
  const [incomeThreshold, setIncomeThreshold] = useState(20);
  const [lowIncomeEducThreshold, setLowIncomeEducThreshold] = useState(12);
  const [highIncomeEducThreshold, setHighIncomeEducThreshold] = useState(16);

  // Simulation / "Test Student" Parameters
  const [simIncome, setSimIncome] = useState<number>(25);
  const [simEduc, setSimEduc] = useState<number>(14);
  const [isSimulating, setIsSimulating] = useState(false);

  // Copy State
  const [copied, setCopied] = useState(false);

  // Reset Handler
  const handleReset = () => {
    setIncomeThreshold(20);
    setLowIncomeEducThreshold(12);
    setHighIncomeEducThreshold(16);
    setSimIncome(25);
    setSimEduc(14);
    setIsSimulating(false);
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(PYTHON_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Real-time Accuracy Calculation
  const stats = useMemo(() => {
    let correctCount = 0;
    
    INITIAL_DATA.forEach(point => {
      let predictedJob = false;
      
      if (point.income < incomeThreshold) {
        // Low Income Branch
        // If Edu < Threshold -> No Job, Else Job
        predictedJob = point.education >= lowIncomeEducThreshold;
      } else {
        // High Income Branch
        // If Edu < Threshold -> No Job, Else Job
        predictedJob = point.education >= highIncomeEducThreshold;
      }

      if (predictedJob === point.foundJob) {
        correctCount++;
      }
    });

    const accuracy = Math.round((correctCount / INITIAL_DATA.length) * 100);
    return { accuracy, correctCount, total: INITIAL_DATA.length };
  }, [incomeThreshold, lowIncomeEducThreshold, highIncomeEducThreshold]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-12">
      {/* Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <span className="font-bold text-slate-800 text-lg tracking-tight">Econometrics<span className="text-blue-600">Lab</span></span>
          </div>
          <div className="text-xs text-slate-500 hidden sm:block">
            Interactive Decision Tree Visualization
          </div>
        </div>
      </nav>

      {/* Main Controls Toolbar */}
      <div className="bg-white border-b border-slate-200 py-4 px-4 shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row gap-6 items-center justify-center">
          <div className="flex items-center gap-4 w-full md:w-auto">
             <span className="text-xs font-bold text-slate-400 uppercase tracking-wide w-24">Root Split</span>
             <div className="flex flex-col w-full md:w-48">
               <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                 <span>Income</span>
                 <span className="text-blue-600">${incomeThreshold}k</span>
               </div>
               <input 
                  type="range" min="10" max="60" value={incomeThreshold} 
                  onChange={(e) => setIncomeThreshold(Number(e.target.value))}
                  className="h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
             </div>
          </div>

          <div className="w-px h-8 bg-slate-200 hidden md:block"></div>

          <div className="flex items-center gap-4 w-full md:w-auto">
             <span className="text-xs font-bold text-slate-400 uppercase tracking-wide w-24">Leaf Splits</span>
             
             <div className="flex flex-col w-full md:w-48">
               <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                 <span>Edu (Low Inc)</span>
                 <span className="text-blue-600">{lowIncomeEducThreshold}y</span>
               </div>
               <input 
                  type="range" min="8" max="16" value={lowIncomeEducThreshold} 
                  onChange={(e) => setLowIncomeEducThreshold(Number(e.target.value))}
                  className="h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
             </div>

             <div className="flex flex-col w-full md:w-48 ml-4">
               <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                 <span>Edu (High Inc)</span>
                 <span className="text-blue-600">{highIncomeEducThreshold}y</span>
               </div>
               <input 
                  type="range" min="12" max="18" value={highIncomeEducThreshold} 
                  onChange={(e) => setHighIncomeEducThreshold(Number(e.target.value))}
                  className="h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
             </div>
          </div>

          <div className="w-px h-8 bg-slate-200 hidden md:block"></div>

          <button 
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-slate-500 bg-slate-50 hover:bg-slate-100 hover:text-blue-600 rounded-lg border border-slate-200 transition-all shadow-sm active:scale-95"
            title="Reset to default values"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            RESET
          </button>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-8">

        {/* Introduction Section */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-bold text-slate-800 mb-3">Understanding Decision Trees in Econometrics</h2>
          <div className="text-slate-600 text-sm leading-relaxed space-y-4">
            <p>
              A decision tree is a non-linear model that predicts outcomes by splitting data into subgroups based on specific rules. 
              In this case study, we analyze how <strong>Prior Income</strong> and <strong>Education</strong> interact to predict whether a student finds a job.
              This allows us to see how education impacts job prospects differently depending on a student's background.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <span className="font-bold text-slate-800 block mb-1">1. The Root</span>
                <p className="text-xs">The starting point that splits the entire population based on the most critical feature (e.g., Is Income &lt; $20k?).</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <span className="font-bold text-slate-800 block mb-1">2. Branches</span>
                <p className="text-xs">Intermediate paths where further questions are asked to refine the group (e.g., Is Education &lt; 16 years?).</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <span className="font-bold text-slate-800 block mb-1">3. Leaves</span>
                <p className="text-xs">The final endpoints (colored zones) where a prediction is made: <strong>Job Found</strong> or <strong>No Job</strong>.</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-auto">
           {/* Left: Data View */}
           <div className="h-[500px]">
             <ScatterPlot 
              data={INITIAL_DATA} 
              incomeThreshold={incomeThreshold}
              lowIncomeEducThreshold={lowIncomeEducThreshold}
              highIncomeEducThreshold={highIncomeEducThreshold}
              simulationPoint={isSimulating ? { income: simIncome, education: simEduc } : null}
             />
           </div>

           {/* Right: Logic View */}
           <div className="h-[500px]">
             <DecisionTreeVisualizer 
              incomeThreshold={incomeThreshold}
              lowIncomeEducThreshold={lowIncomeEducThreshold}
              highIncomeEducThreshold={highIncomeEducThreshold}
              simulationPoint={isSimulating ? { income: simIncome, education: simEduc } : null}
             />
           </div>
        </div>

        {/* Bottom Section: Educational Content & Simulator */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-12 gap-6">
           
           {/* Accuracy Card */}
           <div className="md:col-span-4 bg-white p-5 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-slate-800 text-lg mb-2">Model Accuracy</h3>
                <div className="flex items-center gap-3 mb-2">
                    <div className="flex-1 h-3 bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-green-500 rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${stats.accuracy}%` }}
                      ></div>
                    </div>
                    <span className="text-lg font-bold text-slate-700">{stats.accuracy}%</span>
                </div>
                <p className="text-xs text-slate-500">
                  {stats.correctCount} out of {stats.total} students classified correctly.
                </p>
              </div>
              <p className="text-sm text-slate-600 mt-4 leading-relaxed">
                Move the sliders above! Notice how accuracy drops if you set the thresholds poorly. This simulates "training" the model.
              </p>
           </div>

           {/* Simulator Card */}
           <div className="md:col-span-4 bg-blue-50 border border-blue-100 p-5 rounded-xl shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-slate-800 text-lg">Test a Student</h3>
                <label className="flex items-center cursor-pointer">
                  <div className="relative">
                    <input type="checkbox" className="sr-only" checked={isSimulating} onChange={() => setIsSimulating(!isSimulating)} />
                    <div className={`block w-10 h-6 rounded-full transition-colors ${isSimulating ? 'bg-blue-600' : 'bg-slate-300'}`}></div>
                    <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${isSimulating ? 'transform translate-x-4' : ''}`}></div>
                  </div>
                  <span className="ml-2 text-xs font-semibold text-slate-600">{isSimulating ? 'ON' : 'OFF'}</span>
                </label>
              </div>
              
              <div className={`space-y-4 transition-opacity ${isSimulating ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                 <div>
                   <label className="block text-xs font-semibold text-slate-600 mb-1">Income ($k)</label>
                   <input 
                      type="range" min="0" max="100" value={simIncome} 
                      onChange={(e) => setSimIncome(Number(e.target.value))}
                      className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="text-right text-xs font-bold text-blue-700">${simIncome}k</div>
                 </div>
                 <div>
                   <label className="block text-xs font-semibold text-slate-600 mb-1">Education (Years)</label>
                   <input 
                      type="range" min="0" max="20" step="0.5" value={simEduc} 
                      onChange={(e) => setSimEduc(Number(e.target.value))}
                      className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="text-right text-xs font-bold text-blue-700">{simEduc} yrs</div>
                 </div>
                 <p className="text-xs text-blue-800 italic mt-2">
                   Turn this ON to see how a specific student flows through the tree logic.
                 </p>
              </div>
           </div>

           {/* Education Card */}
           <div className="md:col-span-4 bg-white p-5 rounded-xl shadow-sm border border-slate-200">
              <h3 className="font-bold text-slate-800 text-lg mb-2">Interaction Effects</h3>
              <p className="text-sm text-slate-600 leading-relaxed mb-4">
                The tree structure explicitly models an interaction: The effect of Education is <strong>conditional</strong> on Income. 
              </p>
              <div className="bg-slate-50 p-3 rounded-lg text-xs text-slate-500 border border-slate-100">
                 Notice how the education threshold is lower ({lowIncomeEducThreshold}y) for the low-income group compared to high-income ({highIncomeEducThreshold}y).
              </div>
           </div>
        </div>

        {/* Google Colab Section */}
        <div className="mt-12 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="bg-slate-800 p-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-xl font-bold text-white mb-1">Run the code in Google Colab</h2>
                    <p className="text-slate-400 text-sm">Analyze this case study using Python and Scikit-Learn.</p>
                </div>
                
                <div className="flex gap-3">
                  <a 
                    href="https://colab.research.google.com/notebooks/empty.ipynb"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 border border-slate-600"
                  >
                     <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                     </svg>
                     Open Colab
                  </a>
                  
                  <button 
                    onClick={handleCopyCode}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2"
                  >
                    {copied ? (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        Copied!
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        Copy Code
                      </>
                    )}
                  </button>
                </div>
            </div>
            <div className="p-0 bg-[#1e1e1e] overflow-x-auto relative">
                <div className="absolute top-0 right-0 p-2 text-xs text-slate-500 font-mono">Python</div>
                <pre className="p-6 text-sm font-mono text-slate-300 leading-relaxed overflow-x-auto">
                    {PYTHON_CODE}
                </pre>
            </div>
        </div>

      </main>
    </div>
  );
};

export default App;