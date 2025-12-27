
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { DataPoint } from '../types';
import { COLORS } from '../constants';

interface ScatterPlotProps {
  data: DataPoint[];
  incomeThreshold: number;
  lowIncomeEducThreshold: number;
  highIncomeEducThreshold: number;
  simulationPoint?: { income: number; education: number } | null;
}

const ScatterPlot: React.FC<ScatterPlotProps> = ({ 
  data, 
  incomeThreshold, 
  lowIncomeEducThreshold, 
  highIncomeEducThreshold,
  simulationPoint
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    // Responsive dimensions
    const containerWidth = containerRef.current.clientWidth;
    const margin = { top: 30, right: 30, bottom: 50, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear()
      .domain([0, 20])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0]);

    // Gridlines
    const makeXGridlines = () => d3.axisBottom(x).ticks(10);
    const makeYGridlines = () => d3.axisLeft(y).ticks(10);

    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(makeXGridlines().tickSize(-height).tickFormat(() => ''))
      .style('stroke-opacity', 0.1)
      .style('color', COLORS.grid);

    svg.append('g')
      .attr('class', 'grid')
      .call(makeYGridlines().tickSize(-width).tickFormat(() => ''))
      .style('stroke-opacity', 0.1)
      .style('color', COLORS.grid);

    // Background Zones
    const zones = svg.append('g').attr('class', 'zones');

    // 1. High Income, High Education (Blue)
    zones.append('rect')
      .attr('x', x(highIncomeEducThreshold))
      .attr('y', 0)
      .attr('width', Math.max(0, width - x(highIncomeEducThreshold)))
      .attr('height', y(incomeThreshold))
      .attr('fill', COLORS.zoneBlue)
      .attr('opacity', 0.8);

    // 2. High Income, Low Education (Pink)
    zones.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', x(highIncomeEducThreshold))
      .attr('height', y(incomeThreshold))
      .attr('fill', COLORS.zonePink)
      .attr('opacity', 0.8);

    // 3. Low Income, High Education (Green)
    zones.append('rect')
      .attr('x', x(lowIncomeEducThreshold))
      .attr('y', y(incomeThreshold))
      .attr('width', Math.max(0, width - x(lowIncomeEducThreshold)))
      .attr('height', Math.max(0, height - y(incomeThreshold)))
      .attr('fill', COLORS.zoneGreen)
      .attr('opacity', 0.8);

    // 4. Low Income, Low Education (Orange)
    zones.append('rect')
      .attr('x', 0)
      .attr('y', y(incomeThreshold))
      .attr('width', x(lowIncomeEducThreshold))
      .attr('height', Math.max(0, height - y(incomeThreshold)))
      .attr('fill', COLORS.zoneOrange)
      .attr('opacity', 0.8);

    // Split Lines (Dashed)
    const splitGroup = svg.append('g').attr('class', 'splits');

    // Horizontal Income Line
    splitGroup.append('line')
      .attr('x1', 0)
      .attr('y1', y(incomeThreshold))
      .attr('x2', width)
      .attr('y2', y(incomeThreshold))
      .attr('stroke', COLORS.splitLine)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4');

    splitGroup.append('text')
      .attr('x', width + 5)
      .attr('y', y(incomeThreshold))
      .attr('dy', '0.32em')
      .attr('class', 'text-xs font-bold fill-slate-500')
      .text(`$${incomeThreshold}k`);

    // Top Vertical Line (High Income Edu Split)
    splitGroup.append('line')
      .attr('x1', x(highIncomeEducThreshold))
      .attr('y1', 0)
      .attr('x2', x(highIncomeEducThreshold))
      .attr('y2', y(incomeThreshold))
      .attr('stroke', COLORS.splitLine)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4');
    
    // Bottom Vertical Line (Low Income Edu Split)
    splitGroup.append('line')
      .attr('x1', x(lowIncomeEducThreshold))
      .attr('y1', y(incomeThreshold))
      .attr('x2', x(lowIncomeEducThreshold))
      .attr('y2', height)
      .attr('stroke', COLORS.splitLine)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4');

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(10).tickSize(5).tickPadding(10))
      .attr('class', 'text-xs text-slate-500')
      .select('.domain').remove(); // Remove solid black line

    svg.append('g')
      .call(d3.axisLeft(y).ticks(10).tickSize(5).tickPadding(10))
      .attr('class', 'text-xs text-slate-500')
      .select('.domain').remove();

    // Axis Labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm font-semibold fill-slate-700')
      .text('Education (Years)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm font-semibold fill-slate-700')
      .text('Prior Income (k$)');

    // Data Points
    svg.selectAll('.dot')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.education))
      .attr('cy', d => y(d.income))
      .attr('r', 6)
      .attr('fill', d => d.foundJob ? COLORS.pointFound : COLORS.pointNotFound)
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .style('opacity', simulationPoint ? 0.4 : 1); // Dim if simulating

    // Simulation Point
    if (simulationPoint) {
      const simGroup = svg.append('g')
        .attr('transform', `translate(${x(simulationPoint.education)}, ${y(simulationPoint.income)})`);
      
      // Pulse Effect
      simGroup.append('circle')
        .attr('r', 15)
        .attr('fill', '#facc15') // Yellow-400
        .attr('opacity', 0.5)
        .append('animate')
        .attr('attributeName', 'r')
        .attr('values', '10;20;10')
        .attr('dur', '1.5s')
        .attr('repeatCount', 'indefinite');

      // The star/dot
      simGroup.append('circle')
        .attr('r', 8)
        .attr('fill', '#ca8a04') // Yellow-700
        .attr('stroke', 'white')
        .attr('stroke-width', 2);
        
      simGroup.append('text')
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('class', 'text-xs font-bold fill-slate-800 bg-white')
        .text('YOU');
    }

  }, [data, incomeThreshold, lowIncomeEducThreshold, highIncomeEducThreshold, simulationPoint]);

  return (
    <div ref={containerRef} className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full">
      <div className="flex justify-between items-center mb-6">
        <h3 className="font-bold text-slate-800 text-lg">Feature Space Partitioning</h3>
        <div className="flex gap-4 text-xs font-medium bg-slate-50 px-3 py-2 rounded-full border border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{backgroundColor: COLORS.pointFound}}></div>
            <span>Found Job</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{backgroundColor: COLORS.pointNotFound}}></div>
            <span>No Job</span>
          </div>
        </div>
      </div>
      <div className="w-full flex justify-center">
         <svg ref={svgRef} className="overflow-visible"></svg>
      </div>
    </div>
  );
};

export default ScatterPlot;
