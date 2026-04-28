interface Props {
  title: string;
  value: string;
  subtitle?: string;
}

export default function KpiCard({ title, value, subtitle }: Props) {
  return (
    <div className="bg-white rounded-lg shadow p-4 border">
      <p className="text-sm text-gray-500">{title}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
    </div>
  );
}
