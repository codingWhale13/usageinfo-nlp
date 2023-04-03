export { default } from "next-auth/middleware";

export const config = {
  matcher: ["/browse/:path*", "/view/:path*", "/api/upload/:path*"],
};
