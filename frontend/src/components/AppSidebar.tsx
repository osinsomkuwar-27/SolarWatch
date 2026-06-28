import { Link, useRouterState } from "@tanstack/react-router";
import { Activity, History, BarChart3, Satellite, Sun } from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

const items = [
  { title: "Live Dashboard", url: "/", icon: Activity },
  { title: "History", url: "/history", icon: History },
  { title: "Model Insights", url: "/insights", icon: BarChart3 },
  { title: "Aditya-L1", url: "/aditya", icon: Satellite },
] as const;

export function AppSidebar() {
  const currentPath = useRouterState({ select: (s) => s.location.pathname });
  const isActive = (path: string) =>
    path === "/" ? currentPath === "/" : currentPath.startsWith(path);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border px-4 py-5">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-sky/10 ring-1 ring-sky/30 text-sky">
            <Sun className="h-5 w-5" />
          </div>
          <div className="flex flex-col leading-tight group-data-[collapsible=icon]:hidden">
            <span className="font-serif text-lg text-foreground">SolarWatch</span>
            <span className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
              Forecast Console
            </span>
          </div>
        </Link>
      </SidebarHeader>

      <SidebarContent className="px-2 pt-4">
        <SidebarGroup>
          <SidebarGroupLabel className="px-3 font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
            Mission
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => {
                const active = isActive(item.url);
                return (
                  <SidebarMenuItem key={item.url}>
                    <SidebarMenuButton asChild isActive={active} tooltip={item.title}>
                      <Link
                        to={item.url}
                        className="flex items-center gap-3 rounded-md px-3 py-2 text-sm"
                      >
                        <item.icon className="h-4 w-4 shrink-0" />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border p-4 group-data-[collapsible=icon]:hidden">
        <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
          Data Source
        </div>
        <div className="mt-1 text-sm text-text-muted">
          NOAA SWPC · GOES-18
        </div>
        <div className="mt-3 flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full rounded-full bg-safe opacity-60 animate-ping" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-safe" />
          </span>
          <span className="text-sm text-safe">Feed live</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
