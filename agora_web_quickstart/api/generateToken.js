import { RtcTokenBuilder, RtcRole } from "agora-access-token";

export default function handler(req, res) {
  const appID = process.env.AGORA_APP_ID;
  const appCertificate = process.env.AGORA_APP_CERTIFICATE;

  const channelName = req.query.channel;
  const uid = req.query.uid || 0;
  const role = RtcRole.PUBLISHER;
  const expireTime = 3600;

  const token = RtcTokenBuilder.buildTokenWithUid(
    appID,
    appCertificate,
    channelName,
    uid,
    role,
    expireTime
  );

  res.status(200).json({ token });
}
