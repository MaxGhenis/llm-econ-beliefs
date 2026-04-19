"use client";

import { getModelLabel } from "@/lib/model-meta";
import type {
  IntervalMethodDefinition,
  IntervalSnapshot,
  ModelSummary,
} from "@/lib/dashboard-types";

interface IntervalPlotProps {
  models: ModelSummary[];
  method: IntervalMethodDefinition;
}

const LEFT_GUTTER = 192;
const RIGHT_GUTTER = 72;
const TOP_GUTTER = 32;
const ROW_HEIGHT = 64;
const CHART_WIDTH = 960;

export function IntervalPlot({ models, method }: IntervalPlotProps) {
  const rows = models
    .map((model) => ({
      model,
      interval: model.intervals[method.id],
    }))
    .filter(
      (row) =>
        row.interval.center !== null ||
        row.interval.lower !== null ||
        row.interval.upper !== null,
    );

  if (!rows.length) {
    return (
      <div
        className="rounded-lg border px-6 py-8 text-center text-xs"
        style={{
          background: "var(--bg-surface)",
          borderColor: "var(--border)",
          color: "var(--text-tertiary)",
        }}
      >
        No interval data available for this method.
      </div>
    );
  }

  const allValues = rows.flatMap(({ interval }) =>
    [interval.lower, interval.center, interval.upper].filter(
      (value): value is number => value !== null,
    ),
  );
  const plotWidth = CHART_WIDTH - LEFT_GUTTER - RIGHT_GUTTER;
  const chartHeight = TOP_GUTTER + rows.length * ROW_HEIGHT + 8;
  const ticks = buildNiceTicks(
    Math.min(...allValues, 0),
    Math.max(...allValues, 0),
    6,
  );
  const domainMin = ticks[0] ?? Math.min(...allValues, 0);
  const domainMax = ticks[ticks.length - 1] ?? Math.max(...allValues, 0);
  const zeroX =
    scaleValue(0, domainMin, domainMax, plotWidth) + LEFT_GUTTER;

  return (
    <div
      className="rounded-lg border"
      style={{
        background: "var(--bg-surface)",
        borderColor: "var(--border)",
      }}
    >
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${chartHeight}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${method.label} comparison across models`}
      >
        <defs>
          {/* Interval gradient: gold to blue */}
          <linearGradient id="interval-bar" x1="0%" x2="100%">
            <stop offset="0%" stopColor="var(--gold)" stopOpacity="0.6" />
            <stop offset="50%" stopColor="var(--gold)" stopOpacity="0.8" />
            <stop offset="100%" stopColor="var(--gold)" stopOpacity="0.6" />
          </linearGradient>

          {/* Glow filter for center dot */}
          <filter id="dot-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>

          {/* Subtle glow for the bar */}
          <filter id="bar-glow" x="-20%" y="-200%" width="140%" height="500%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Zero reference line */}
        <line
          x1={zeroX}
          x2={zeroX}
          y1={TOP_GUTTER - 4}
          y2={chartHeight - 4}
          stroke="var(--blue)"
          strokeOpacity="0.6"
          strokeWidth="1.5"
        />

        {/* Tick lines and labels */}
        {ticks.map((tick) => {
          const x =
            scaleValue(tick, domainMin, domainMax, plotWidth) + LEFT_GUTTER;
          const isZero = Math.abs(tick) < 1e-10;
          return (
            <g key={tick}>
              <line
                x1={x}
                x2={x}
                y1={TOP_GUTTER - 4}
                y2={chartHeight - 4}
                stroke={isZero ? "transparent" : "var(--border)"}
                strokeDasharray="2 8"
              />
              <text
                x={x}
                y={16}
                textAnchor="middle"
                fill={isZero ? "var(--blue)" : "var(--text-tertiary)"}
                fontSize="10"
                fontFamily="var(--font-jetbrains), monospace"
              >
                {formatTickNumber(tick)}
              </text>
            </g>
          );
        })}

        {/* Data rows */}
        {rows.map(({ model, interval }, index) => {
          const y = TOP_GUTTER + index * ROW_HEIGHT + ROW_HEIGHT / 2;
          const lower =
            interval.lower !== null
              ? scaleValue(interval.lower, domainMin, domainMax, plotWidth) +
                LEFT_GUTTER
              : null;
          const upper =
            interval.upper !== null
              ? scaleValue(interval.upper, domainMin, domainMax, plotWidth) +
                LEFT_GUTTER
              : null;
          const center =
            interval.center !== null
              ? scaleValue(interval.center, domainMin, domainMax, plotWidth) +
                LEFT_GUTTER
              : null;

          return (
            <g key={model.modelName}>
              {/* Row background on hover area */}
              {index > 0 && (
                <line
                  x1={0}
                  x2={CHART_WIDTH}
                  y1={y - ROW_HEIGHT / 2}
                  y2={y - ROW_HEIGHT / 2}
                  stroke="var(--border)"
                  strokeOpacity="0.5"
                />
              )}

              {/* Model label */}
              <text
                x={14}
                y={y - 4}
                fill="var(--text-primary)"
                fontSize="13"
                fontFamily="var(--font-fraunces), serif"
                fontWeight="600"
              >
                {getModelLabel(model.modelName)}
              </text>
              <text
                x={14}
                y={y + 12}
                fill="var(--text-tertiary)"
                fontSize="10"
                fontFamily="var(--font-jetbrains), monospace"
              >
                {model.nSuccessfulRuns} runs
              </text>

              {/* Faint baseline */}
              <line
                x1={LEFT_GUTTER}
                x2={CHART_WIDTH - RIGHT_GUTTER}
                y1={y}
                y2={y}
                stroke="var(--border)"
                strokeOpacity="0.4"
              />

              {/* Interval bar */}
              {lower !== null && upper !== null && (
                <>
                  {/* Glow layer */}
                  <line
                    x1={lower}
                    x2={upper}
                    y1={y}
                    y2={y}
                    stroke="var(--gold)"
                    strokeOpacity="0.15"
                    strokeLinecap="round"
                    strokeWidth="20"
                  />
                  {/* Main bar */}
                  <line
                    x1={lower}
                    x2={upper}
                    y1={y}
                    y2={y}
                    stroke="url(#interval-bar)"
                    strokeLinecap="round"
                    strokeWidth="6"
                    filter="url(#bar-glow)"
                  />
                  {/* Whisker caps */}
                  <line
                    x1={lower}
                    x2={lower}
                    y1={y - 10}
                    y2={y + 10}
                    stroke="var(--gold)"
                    strokeWidth="1.5"
                    strokeOpacity="0.7"
                  />
                  <line
                    x1={upper}
                    x2={upper}
                    y1={y - 10}
                    y2={y + 10}
                    stroke="var(--gold)"
                    strokeWidth="1.5"
                    strokeOpacity="0.7"
                  />
                </>
              )}

              {/* Center dot */}
              {center !== null && (
                <>
                  {/* Outer glow */}
                  <circle
                    cx={center}
                    cy={y}
                    r={10}
                    fill="var(--gold)"
                    opacity="0.08"
                  />
                  {/* Core */}
                  <circle
                    cx={center}
                    cy={y}
                    r={5}
                    fill="var(--gold)"
                    filter="url(#dot-glow)"
                  />
                  {/* Inner bright core */}
                  <circle
                    cx={center}
                    cy={y}
                    r={2}
                    fill="#fff"
                    opacity="0.6"
                  />
                </>
              )}

              {/* Right-side value label */}
              <text
                x={CHART_WIDTH - 8}
                y={y + 4}
                textAnchor="end"
                fill="var(--text-secondary)"
                fontSize="11"
                fontFamily="var(--font-jetbrains), monospace"
              >
                {formatIntervalLabel(interval)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function buildNiceTicks(min: number, max: number, targetCount: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  if (min === max) return [min];

  const roughStep = Math.abs(max - min) / Math.max(targetCount - 1, 1);
  const step = snap125(roughStep);
  const start = Math.floor(min / step) * step;
  const stop = Math.ceil(max / step) * step;
  const ticks: number[] = [];

  for (let value = start; value <= stop + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(12)));
  }

  return ticks;
}

function scaleValue(
  value: number,
  min: number,
  max: number,
  width: number,
): number {
  if (max <= min) return width / 2;
  return ((value - min) / (max - min)) * width;
}

function formatIntervalLabel(interval: IntervalSnapshot): string {
  if (
    interval.center === null ||
    interval.lower === null ||
    interval.upper === null
  )
    return "—";
  return `${formatNumber(interval.center)} [${formatNumber(interval.lower)}, ${formatNumber(interval.upper)}]`;
}

function formatNumber(value: number): string {
  const abs = Math.abs(value);
  const fractionDigits = abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value);
}

function formatTickNumber(value: number): string {
  if (Math.abs(value) < 1e-10) return "0";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 3,
  }).format(value);
}

function snap125(value: number): number {
  if (value <= 0 || !Number.isFinite(value)) return 1;
  const exponent = Math.floor(Math.log10(value));
  const scale = 10 ** exponent;
  const normalized = value / scale;
  if (normalized <= 1) return scale;
  if (normalized <= 2) return 2 * scale;
  if (normalized <= 5) return 5 * scale;
  return 10 * scale;
}
