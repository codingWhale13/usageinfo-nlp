// pages/api/auth/[...nextauth].js
import NextAuth from "next-auth";
import GitHubProvider from "next-auth/providers/github";

export default NextAuth({
  secret: process.env.NEXTAUTH_SECRET,
  debug: true,
  providers: [
    GitHubProvider({
      clientId: '9e79722d0209e0d25bae',
      clientSecret: 'a862b516afd9188633d2c884238452ea6326aa4a'
    }),
    {
      id: "hpiopenidconnect",
      name: "HPI OpenID Conntect",
      type: "oauth",
      clientId: "7db572d2-244b-48b8-8eb1-9ba41b9099cf",
      clientSecret:
        "a2bc9fffe205e13ba00c907b07229203470e8ef75c437f885239b44a1bdddce563c1afa99d947c2d1f287bcad20bf293119ba91b072d0dcf74ff0450d06af8bb",
      idToken: false,
      //wellKnown: "https://oidc.hpi.de/.well-known/openid-configuration",
      authorization: {
        url: 'https://oidc.hpi.de/auth',
        params: {scope: 'openid email profile'}
      },
      token: {
        url: 'https://oidc.hpi.de/token'
      },
      userinfo: {
        url: "https://oidc.hpi.de/me",
      },
      checks: ['nonce', 'state'],
      profile(profile) {
        console.log(profile);
        return {
          name: profile.name,
          email: profile.email
        };
      },
    },
  ],
});
