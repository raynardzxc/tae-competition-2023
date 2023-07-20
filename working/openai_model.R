# Load Package
source("working/usePackages.r")
pkgnames <- c("openai")
loadPkgs(pkgnames)

# Function to classify tweet sentiment
# $0.1200 / 1K tokens
classify_tweet_sentiment <- function(tweet) {
  
  # Define prompt
  prompt <- paste("You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative. This is the tweet:\n\"", tweet, "\"\n\nSentiment of this tweet is:")
  
  # Make the API request
  response <- create_completion(
    model = "text-davinci-002", # replace with appropriate model
    prompt = prompt,
    max_tokens = 60,
    temperature = 0.7, # feel free to adjust these parameters as needed
    top_p = 1,
    n = 1
  )
  
  sentiment <- response$choices$text
  
  return(sentiment)
}

# Function to classify tweet sentiment
# Model	Input	Output
# 4K context	$0.0015 / 1K tokens	$0.002 / 1K tokens
# 16K context	$0.003 / 1K tokens	$0.004 / 1K tokens
classify_tweet_sentiment_gpt <- function(tweet) {
  
  # Define messages
  messages <- list(
    list("role" = "system", "content" = "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative."),
    list("role" = "user", "content" = tweet)
  )
  
  # Make the API request
  response <- create_chat_completion(
    model = "gpt-3.5-turbo", # replace with appropriate model
    messages = messages,
    max_tokens = 60,
    temperature = 0.7, # feel free to adjust these parameters as needed
    top_p = 1,
    n = 1
  )
  
  # Extract the sentiment from the API response
  sentiment <- response$choices$message.content
  
  return(sentiment)
}

# Example usage
tweet <- "I love this product, it works perfectly!"
print(classify_tweet_sentiment(tweet))