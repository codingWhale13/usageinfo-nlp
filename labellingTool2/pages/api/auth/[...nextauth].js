// pages/api/auth/[...nextauth].js
import NextAuth from "next-auth";
import GitHubProvider from "next-auth/providers/github";
import { ALLOWED_GITHUB_LOGINS } from "../../../utils/allowedUsers";

export default NextAuth({
  debug: true,
  providers: [
    GitHubProvider({
      clientId: process.env.NEXTGITHUB_ID,
      clientSecret: process.env.NEXTGITHUB_SECRET,
    }),
  ],
  callbacks: {
    async signIn({ user, account, profile, email, credentials }) {
      let isAllowedToSignIn = false;
      if (account.provider === "github") {
        isAllowedToSignIn = ALLOWED_GITHUB_LOGINS.includes(profile.login);
      }

      return isAllowedToSignIn;
    },
  },
});
