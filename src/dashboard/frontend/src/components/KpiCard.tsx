interface Props {
  title: string;
  value: string;
  subtitle?: string;
}

export default function KpiCard({ title, value, subtitle }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">
        {title}
      </p>
      <p className="text-2xl font-semibold text-slate-900 mt-1.5">{value}</p>
      {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
    </div>
  );
}
