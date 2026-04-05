import { DashboardClient } from "@/components/dashboard-client";
import { loadDashboardSummaryData } from "@/lib/dashboard-data";

export const dynamic = "force-dynamic";

export default function Home() {
  const data = loadDashboardSummaryData();

  return <DashboardClient data={data} />;
}
