"use client";

import { usePathname } from "next/navigation";
import Sidebar from "@/components/Sidebar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const bare = pathname === "/login" || pathname === "/login/";

  if (bare) {
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="ml-64 flex-1 overflow-auto bg-[#f5f7fb]">{children}</main>
    </div>
  );
}
