import logging
from typing import Dict, Any, List, Tuple

from framework.activity_decorator import activity, ActivityBase, ActivityResult
from framework.api_management import api_manager
from framework.memory import Memory
from skills.skill_chat import chat_skill
from skills.skill_generate_image import ImageGenerationSkill
from skills.skill_x_api import XAPISkill
from dotenv import load_dotenv  # Add this if not already present

logger = logging.getLogger(__name__)

# At the start of your file or in __init__
load_dotenv()  # Add this line to load the .env file


@activity(
    name="post_a_tweet",
    energy_cost=0.4,
    cooldown=3600,  # 1 hour
    required_skills=["twitter_posting", "image_generation"],
)
class PostTweetActivity(ActivityBase):
    """
    Uses a chat skill (OpenAI) to generate tweet text,
    referencing the character's personality from character_config.
    Checks recent tweets in memory to avoid duplication.
    Posts to Twitter via Composio's "Creation of a post" dynamic action.
    """

    def __init__(self):
        super().__init__()
        self.max_length = 280
        # If you know your Twitter username, you can embed it in the link
        # or fetch it dynamically. Otherwise, substitute accordingly:
        self.twitter_username = "barbara12817659"
        # set this to True if you want to generate an image for the tweet
        self.image_generation_enabled = True
        self.default_size = (1024, 1024)  # Added for image generation
        self.default_format = "png"  # Added for image generation

    async def execute(self, shared_data) -> ActivityResult:
        try:
            logger.info("Starting tweet posting activity...")

            # 1) Initialize the chat skill
            if not await chat_skill.initialize():
                return ActivityResult(
                    success=False, error="Failed to initialize chat skill"
                )

            # 2) Gather personality + recent tweets
            character_config = self._get_character_config(shared_data)
            personality_data = character_config.get("personality", {})
            recent_tweets = self._get_recent_tweets(shared_data, limit=10)

            # 3) Generate tweet text with chat skill
            prompt_text = self._build_chat_prompt(personality_data, recent_tweets)
            chat_response = await chat_skill.get_chat_completion(
                prompt=prompt_text,
                system_prompt="You are an AI that composes tweets with the given personality.",
                max_tokens=100,
            )
            if not chat_response["success"]:
                return ActivityResult(success=False, error=chat_response["error"])

            tweet_text = chat_response["data"]["content"].strip()
            if len(tweet_text) > self.max_length:
                tweet_text = tweet_text[: self.max_length - 3] + "..."

            # 4) Generate an image based on the tweet text
            if self.image_generation_enabled:
                image_prompt, media_urls = await self._generate_image_for_tweet(tweet_text, personality_data)
            else:
                image_prompt, media_urls = None, []

            # 5) Post the tweet via X API
            x_api = XAPISkill({
                "enabled": True,
                "twitter_username": self.twitter_username
            })
            post_result = await x_api.post_tweet(tweet_text, media_urls)
            if not post_result["success"]:
                error_msg = post_result.get(
                    "error", "Unknown error posting tweet via Composio"
                )
                logger.error(f"Tweet posting failed: {error_msg}")
                return ActivityResult(success=False, error=error_msg)

            tweet_id = post_result.get("tweet_id")
            tweet_link = (
                f"https://twitter.com/{self.twitter_username}/status/{tweet_id}"
                if tweet_id
                else None
            )

            # 6) Return success, adding link & prompt in metadata
            logger.info(f"Successfully posted tweet: {tweet_text[:50]}...")
            return ActivityResult(
                success=True,
                data={"tweet_id": tweet_id, "content": tweet_text},
                metadata={
                    "length": len(tweet_text),
                    "method": "composio",
                    "model": chat_response["data"].get("model"),
                    "finish_reason": chat_response["data"].get("finish_reason"),
                    "tweet_link": tweet_link,
                    "prompt_used": prompt_text,
                    "image_prompt_used": image_prompt,
                    "image_count": len(media_urls),
                },
            )

        except Exception as e:
            logger.error(f"Failed to post tweet: {e}", exc_info=True)
            return ActivityResult(success=False, error=str(e))

    def _get_character_config(self, shared_data) -> Dict[str, Any]:
        """
        Retrieve character_config from SharedData['system'] or re-init the Being if not found.
        """
        system_data = shared_data.get_category_data("system")
        maybe_config = system_data.get("character_config")
        if maybe_config:
            return maybe_config

        # fallback
        from framework.main import DigitalBeing

        being = DigitalBeing()
        being.initialize()
        return being.configs.get("character_config", {})

    def _get_recent_tweets(self, shared_data, limit: int = 10) -> List[str]:
        """
        Fetch the last N tweets posted (activity_type='PostTweetActivity') from memory.
        """
        system_data = shared_data.get_category_data("system")
        memory_obj: Memory = system_data.get("memory_ref")

        if not memory_obj:
            from framework.main import DigitalBeing

            being = DigitalBeing()
            being.initialize()
            memory_obj = being.memory

        recent_activities = memory_obj.get_recent_activities(limit=50, offset=0)
        tweets = []
        for act in recent_activities:
            if act.get("activity_type") == "PostTweetActivity" and act.get("success"):
                tweet_body = act.get("data", {}).get("content", "")
                if tweet_body:
                    tweets.append(tweet_body)

        return tweets[:limit]

    def _build_chat_prompt(
        self, personality: Dict[str, Any], recent_tweets: List[str]
    ) -> str:
        """
        Construct the user prompt referencing personality + last tweets.
        """
        trait_lines = [f"{t}: {v}" for t, v in personality.items()]
        personality_str = "\n".join(trait_lines)

        if recent_tweets:
            last_tweets_str = "\n".join(f"- {txt}" for txt in recent_tweets)
        else:
            last_tweets_str = "(No recent tweets)"

        return (
            f"Our digital being has these personality traits:\n"
            f"{personality_str}\n\n"
            f"Here are recent tweets:\n"
            f"{last_tweets_str}\n\n"
            f"Write a new short tweet (under 280 chars), consistent with the above, "
            f"but not repeating old tweets. Avoid hashtags or repeated phrases.\n"
        )

    def _build_image_prompt(self, tweet_text: str, personality: Dict[str, Any]) -> str:
        personality_str = "\n".join(f"{t}: {v}" for t, v in personality.items())
        return f"Our digital being has these personality traits:\n" \
               f"{personality_str}\n\n" \
               f"And is creating a tweet with the text: {tweet_text}\n\n" \
               f"Generate an image that represents the story of the tweet and reflects the personality traits. Do not include the tweet text in the image."

    async def _generate_image_for_tweet(self, tweet_text: str, personality_data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Generate an image for the tweet, mint it as an NFT via Crossmint, and upload it to Twitter.
        Returns a tuple of (image_prompt, media_urls).
        If generation fails, returns (None, []).
        """
        logger.info("Decided to generate an image for tweet")
        image_skill = ImageGenerationSkill({
            "enabled": True,
            "max_generations_per_day": 50,
            "supported_formats": ["png", "jpg"],
        })

        if await image_skill.can_generate():
            image_prompt = self._build_image_prompt(tweet_text, personality_data)
            image_result = await image_skill.generate_image(
                prompt=image_prompt,
                size=self.default_size,
                format=self.default_format
            )
            
            if image_result.get("success") and image_result.get("image_data", {}).get("url"):
                image_url = image_result["image_data"]["url"]
                
                # Mint NFT using Crossmint API
                try:
                    crossmint_result = await self._mint_nft_with_crossmint(
                        image_url=image_url,
                        tweet_text=tweet_text,
                        personality_data=personality_data
                    )
                    logger.info(f"Successfully minted NFT: {crossmint_result}")
                except Exception as e:
                    logger.error(f"Failed to mint NFT: {e}")
                
                return image_prompt, [image_url]
        else:
            logger.warning("Image generation not available, proceeding with text-only tweet")
        
        return None, []

    async def _mint_nft_with_crossmint(self, image_url: str, tweet_text: str, personality_data: Dict[str, Any]) -> Dict:
        """
        Mint an NFT using the Crossmint API with Story protocol
        """
        import aiohttp
        from datetime import datetime
        import os

        logger.info(f"Starting NFT minting with image_url: {image_url}")
        
        API_KEY = os.environ.get("CROSSMINT_STAGING_API_KEY")
        ENV = "staging"
        # RECIPIENT_EMAIL = "mrcleantoiletdao@gmail.com"
        RECIPIENT_EMAIL = os.environ.get("RECEIPIENT_ID")
        COLLECTION_ID = os.environ.get("CROSSMINT_COLLECTION_ID")

        url = f"https://{ENV}.crossmint.com/api/v1/ip/collections/{COLLECTION_ID}/ipassets"
        
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        safe_name = f"Tweet NFT {datetime.now().strftime('%m%d-%H%M')}"

        # Prepare metadata for Story protocol
        payload = {
            "owner": f"email:{RECIPIENT_EMAIL}:story-testnet",
            "reuploadLinkedFiles": True,
            "nftMetadata": {
                "name": safe_name,
                "description": tweet_text,
                "image": image_url
            },
            "ipAssetMetadata": {
                "title": safe_name,
                "createdAt": timestamp,
                "ipType": "social-media",
                "creators": [
                    {
                        "name": "AI Digital Being",
                        "email": RECIPIENT_EMAIL,
                        "crossmintUserLocator": f"email:{RECIPIENT_EMAIL}:story-testnet",
                        "description": "AI Content Creator",
                        "contributionPercent": 100
                    }
                ],
                "attributes": [
                    {
                        "key": "Platform",
                        "value": "Twitter"
                    }
                ]
            },
            "sendNotification": True,
            "locale": "en-US",
            "licenseTerms": [{"type": "non-commercial-social-remixing"}]
        }

        # Add personality traits as attributes
        if personality_data:
            for trait, value in personality_data.items():
                payload["ipAssetMetadata"]["attributes"].append({
                    "key": trait,
                    "value": str(value)
                })

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": API_KEY
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                result = await response.json()
                if response.status != 200:
                    logger.error(f"NFT Minting Failed - Status: {response.status}")
                    logger.error(f"Response Body: {result}")
                    logger.error(f"Request URL: {url}")
                    safe_headers = {k: v for k, v in headers.items() if k.lower() != 'x-api-key'}
                    logger.error(f"Request Headers: {safe_headers}")
                    raise Exception(f"Failed to mint NFT - Status {response.status}: {result.get('message', 'No error message provided')}")
                logger.info(f"Successfully minted NFT")
                return result
