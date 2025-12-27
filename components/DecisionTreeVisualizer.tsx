
import React from 'react';
import { COLORS } from '../constants';

interface TreeVisualizerProps {
  incomeThreshold: number;
  lowIncomeEducThreshold: number;
  highIncomeEducThreshold: number;
  simulationPoint?: { income: number; education: number } | null;
}

const DecisionTreeVisualizer: React.FC<TreeVisualizerProps> = ({
  incomeThreshold,
  lowIncomeEducThreshold,
  highIncomeEducThreshold,
  simulationPoint
}) => {
  
  // Logic to determine active path
  const isHighIncome = simulationPoint && simulationPoint.income >= incomeThreshold;
  const isLowIncome = simulationPoint && simulationPoint.income < incomeThreshold;

  // High Income Branch Logic
  const isHighInc_HighEdu = isHighIncome && simulationPoint!.education >= highIncomeEducThreshold;
  const isHighInc_LowEdu = isHighIncome && simulationPoint!.education < highIncomeEducThreshold;

  // Low Income Branch Logic
  const isLowInc_HighEdu = isLowIncome && simulationPoint!.education >= lowIncomeEducThreshold;
  const isLowInc_LowEdu = isLowIncome && simulationPoint!.education < lowIncomeEducThreshold;

  // Helper for Node Styling
  const getNodeOpacity = (active: boolean | undefined) => {
    if (!simulationPoint) return 'opacity-100'; 
    return active ? 'opacity-100 ring-2 ring-yellow-400 ring-offset-2' : 'opacity-30 grayscale';
  };
  
  // Helper for Connector Line Styling
  const getLineClass = (active: boolean | undefined) => {
    const base = "absolute w-1/2 h-6 border-slate-300 transition-all duration-300";
    if (!simulationPoint) return `${base} border-slate-300 opacity-100`;
    return active 
      ? `${base} border-yellow-500 shadow-[0_0_10px_rgba(234,179,8,0.5)] opacity-100 z-10 border-opacity-100` // Active
      : `${base} border-slate-300 opacity-20`; // Inactive
  };

  const getStemClass = (active: boolean | undefined) => {
     const base = "h-6 w-0.5 bg-slate-300 mx-auto transition-all duration-300";
     if (!simulationPoint) return base;
     return active ? "h-6 w-0.5 bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.8)]" : `${base} opacity-20`;
  }

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col transition-all duration-300">
      <h3 className="font-bold text-slate-800 text-lg mb-4 flex justify-between items-center">
        Decision Tree Logic
        {simulationPoint && <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded font-semibold animate-pulse">Tracing Path...</span>}
      </h3>
      
      <div className="flex-1 flex flex-col items-center justify-start relative">
        
        {/* Level 1: Root */}
        <div className={`relative z-20 transition-all duration-300 ${getNodeOpacity(true)}`}>
          <div className="bg-slate-800 text-white px-6 py-3 rounded-lg shadow-md font-semibold text-sm border-2 border-slate-700">
            Income {'<'} {incomeThreshold}k?
          </div>
        </div>

        {/* Stem Level 1 */}
        <div className={getStemClass(true)}></div>

        {/* Level 2 Container */}
        <div className="flex w-full justify-center">
          
          {/* Left Branch (Income >= Threshold) */}
          <div className="flex-1 flex flex-col items-center relative">
            {/* Connector Line: Center -> Right -> Top */}
            <div className={`${getLineClass(isHighIncome)} border-t-2 border-l-2 rounded-tl-2xl right-0 top-0 translate-y-[-2px]`}></div>

            {/* Content Wrapper with padding to sit below connector */}
            <div className="pt-6 w-full flex flex-col items-center relative z-10">
               {/* Label */}
               <div className={`absolute -top-3 bg-white px-2 text-xs font-bold text-slate-500 whitespace-nowrap z-20 transition-opacity ${!simulationPoint || isHighIncome ? 'opacity-100' : 'opacity-40'}`}>
                 NO (High Inc)
               </div>

               {/* Node */}
               <div className={`bg-white px-4 py-2.5 rounded-lg border-2 border-slate-300 text-slate-700 font-medium text-sm shadow-sm mb-0 transition-all duration-300 ${getNodeOpacity(isHighIncome)}`}>
                  Education {'<'} {highIncomeEducThreshold}?
               </div>

               {/* Stem Level 2 */}
               <div className={getStemClass(isHighIncome)}></div>

               {/* Level 3 Container (Leaves) */}
               <div className="flex w-full justify-center pt-0">
                  {/* Left Leaf (False) -> Blue */}
                  <div className="flex-1 flex flex-col items-center relative">
                    <div className={`${getLineClass(isHighInc_HighEdu)} border-t-2 border-l-2 rounded-tl-xl right-0 top-0 translate-y-[-2px]`}></div>
                    <div className="pt-6 w-full flex flex-col items-center">
                        <div className={`absolute top-3 bg-white px-1 text-[10px] font-bold text-slate-400 z-10 ${!simulationPoint || isHighInc_HighEdu ? 'opacity-100' : 'opacity-40'}`}>NO</div>
                        <div className={`p-3 rounded-lg shadow-sm border text-center w-full max-w-[80px] transition-all duration-300 ${getNodeOpacity(isHighInc_HighEdu)}`}
                              style={{backgroundColor: COLORS.zoneBlue, borderColor: COLORS.borderBlue}}>
                          <div className="text-xs text-slate-500 mb-1">Predict</div>
                          <div className="font-bold text-xs leading-tight" style={{color: COLORS.pointFound}}>Job Found</div>
                        </div>
                    </div>
                  </div>

                  {/* Right Leaf (True) -> Pink */}
                  <div className="flex-1 flex flex-col items-center relative">
                    <div className={`${getLineClass(isHighInc_LowEdu)} border-t-2 border-r-2 rounded-tr-xl left-0 top-0 translate-y-[-2px]`}></div>
                    <div className="pt-6 w-full flex flex-col items-center">
                        <div className={`absolute top-3 bg-white px-1 text-[10px] font-bold text-slate-400 z-10 ${!simulationPoint || isHighInc_LowEdu ? 'opacity-100' : 'opacity-40'}`}>YES</div>
                        <div className={`p-3 rounded-lg shadow-sm border text-center w-full max-w-[80px] transition-all duration-300 ${getNodeOpacity(isHighInc_LowEdu)}`}
                              style={{backgroundColor: COLORS.zonePink, borderColor: COLORS.borderPink}}>
                          <div className="text-xs text-slate-500 mb-1">Predict</div>
                          <div className="font-bold text-xs leading-tight" style={{color: COLORS.pointNotFound}}>No Job</div>
                        </div>
                    </div>
                  </div>
               </div>
            </div>
          </div>

          {/* Right Branch (Income < Threshold) */}
          <div className="flex-1 flex flex-col items-center relative">
            {/* Connector Line: Center -> Left -> Top */}
            <div className={`${getLineClass(isLowIncome)} border-t-2 border-r-2 rounded-tr-2xl left-0 top-0 translate-y-[-2px]`}></div>

            <div className="pt-6 w-full flex flex-col items-center relative z-10">
               {/* Label */}
               <div className={`absolute -top-3 bg-white px-2 text-xs font-bold text-slate-500 whitespace-nowrap z-20 transition-opacity ${!simulationPoint || isLowIncome ? 'opacity-100' : 'opacity-40'}`}>
                 YES (Low Inc)
               </div>

               {/* Node */}
               <div className={`bg-white px-4 py-2.5 rounded-lg border-2 border-slate-300 text-slate-700 font-medium text-sm shadow-sm mb-0 transition-all duration-300 ${getNodeOpacity(isLowIncome)}`}>
                  Education {'<'} {lowIncomeEducThreshold}?
               </div>

               {/* Stem Level 2 */}
               <div className={getStemClass(isLowIncome)}></div>

               {/* Level 3 Container (Leaves) */}
               <div className="flex w-full justify-center pt-0">
                  {/* Left Leaf (False) -> Green */}
                  <div className="flex-1 flex flex-col items-center relative">
                    <div className={`${getLineClass(isLowInc_HighEdu)} border-t-2 border-l-2 rounded-tl-xl right-0 top-0 translate-y-[-2px]`}></div>
                    <div className="pt-6 w-full flex flex-col items-center">
                        <div className={`absolute top-3 bg-white px-1 text-[10px] font-bold text-slate-400 z-10 ${!simulationPoint || isLowInc_HighEdu ? 'opacity-100' : 'opacity-40'}`}>NO</div>
                        <div className={`p-3 rounded-lg shadow-sm border text-center w-full max-w-[80px] transition-all duration-300 ${getNodeOpacity(isLowInc_HighEdu)}`}
                              style={{backgroundColor: COLORS.zoneGreen, borderColor: COLORS.borderGreen}}>
                          <div className="text-xs text-slate-500 mb-1">Predict</div>
                          <div className="font-bold text-xs leading-tight" style={{color: COLORS.pointFound}}>Job Found</div>
                        </div>
                    </div>
                  </div>

                  {/* Right Leaf (True) -> Orange */}
                  <div className="flex-1 flex flex-col items-center relative">
                    <div className={`${getLineClass(isLowInc_LowEdu)} border-t-2 border-r-2 rounded-tr-xl left-0 top-0 translate-y-[-2px]`}></div>
                    <div className="pt-6 w-full flex flex-col items-center">
                        <div className={`absolute top-3 bg-white px-1 text-[10px] font-bold text-slate-400 z-10 ${!simulationPoint || isLowInc_LowEdu ? 'opacity-100' : 'opacity-40'}`}>YES</div>
                        <div className={`p-3 rounded-lg shadow-sm border text-center w-full max-w-[80px] transition-all duration-300 ${getNodeOpacity(isLowInc_LowEdu)}`}
                              style={{backgroundColor: COLORS.zoneOrange, borderColor: COLORS.borderOrange}}>
                          <div className="text-xs text-slate-500 mb-1">Predict</div>
                          <div className="font-bold text-xs leading-tight" style={{color: COLORS.pointNotFound}}>No Job</div>
                        </div>
                    </div>
                  </div>
               </div>
            </div>
          </div>

        </div>

      </div>
      
      <div className="mt-6 pt-4 border-t border-slate-100 flex items-center gap-3 text-xs text-slate-500">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
        </svg>
        <span><strong>Pro Tip:</strong> Toggle "Test a Student" below to trace a path!</span>
      </div>
    </div>
  );
};

export default DecisionTreeVisualizer;
