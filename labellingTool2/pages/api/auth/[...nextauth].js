// pages/api/auth/[...nextauth].js
import NextAuth from "next-auth";
import GitHubProvider from "next-auth/providers/github";
import { ALLOWED_EMAILS } from "../../../utils/allowedUsers";

export default NextAuth({
  debug: true,
  providers: [
    GitHubProvider({
      clientId: process.env.NEXTGITHUB_ID,
      clientSecret: process.env.NEXTGITHUB_SECRET
    })
  ],
  callbacks: {
    async signIn({ user, account, profile, email, credentials }) {
      const isAllowedToSignIn = ALLOWED_EMAILS.includes(profile.email);
      if (isAllowedToSignIn) {
        return true
      } else {
        // Return false to display a default error message
        return false
        // Or you can return a URL to redirect to:
        // return '/unauthorized'
      }
    }
  }
});
